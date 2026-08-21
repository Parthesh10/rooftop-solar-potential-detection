"""Live resource metrics for the training terminal.

Samples GPU utilisation / VRAM / temperature / power and host CPU / RAM, cheaply
enough to call once per training step.

Backends, in order of preference:
  1. ``pynvml`` (nvidia-ml-py) — direct NVML calls, microseconds per sample.
  2. ``torch.cuda.mem_get_info`` — memory only, always available with CUDA.
  3. nothing — every field comes back None and the caller degrades gracefully.

Never shells out to ``nvidia-smi`` per step: spawning a process every iteration
would cost more than the training step itself on a small model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

__all__ = ["SysMon", "Sample", "GpuGovernor", "fmt_bytes"]


def fmt_bytes(n: float | None, unit: str = "G") -> str:
    if n is None:
        return "?"
    div = {"K": 1024, "M": 1024**2, "G": 1024**3}[unit]
    return f"{n / div:.1f}"


@dataclass
class Sample:
    gpu_util: float | None = None      # %
    gpu_mem_used: float | None = None  # bytes
    gpu_mem_total: float | None = None # bytes
    gpu_temp: float | None = None      # deg C
    gpu_power: float | None = None     # W
    gpu_power_cap: float | None = None # W
    torch_alloc: float | None = None   # bytes, allocated by this process
    torch_reserved: float | None = None
    cpu_percent: float | None = None
    ram_used: float | None = None
    ram_total: float | None = None

    def compact(self) -> str:
        """One-line summary for a tqdm postfix."""
        bits = []
        if self.gpu_util is not None:
            bits.append(f"gpu {self.gpu_util:.0f}%")
        if self.gpu_mem_used is not None and self.gpu_mem_total is not None:
            bits.append(
                f"vram {fmt_bytes(self.gpu_mem_used)}/{fmt_bytes(self.gpu_mem_total)}G"
            )
        elif self.torch_reserved is not None:
            bits.append(f"vram {fmt_bytes(self.torch_reserved)}G")
        if self.gpu_temp is not None:
            bits.append(f"{self.gpu_temp:.0f}C")
        if self.gpu_power is not None:
            cap = f"/{self.gpu_power_cap:.0f}" if self.gpu_power_cap else ""
            bits.append(f"{self.gpu_power:.0f}{cap}W")
        if self.cpu_percent is not None:
            bits.append(f"cpu {self.cpu_percent:.0f}%")
        if self.ram_used is not None and self.ram_total is not None:
            bits.append(f"ram {fmt_bytes(self.ram_used)}/{fmt_bytes(self.ram_total)}G")
        return " ".join(bits)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class SysMon:
    """Resource sampler. Construct once, call :meth:`sample` as often as you like."""

    def __init__(self, device_index: int = 0, enabled: bool = True):
        self.enabled = enabled
        self.device_index = device_index
        self._nvml = None
        self._handle = None
        self._psutil = None
        self._torch = None

        if not enabled:
            return

        try:
            import torch

            self._torch = torch if torch.cuda.is_available() else None
        except Exception:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml = pynvml
        except Exception:
            self._nvml = None

        try:
            import psutil

            self._psutil = psutil
            psutil.cpu_percent(interval=None)  # prime the first delta
        except Exception:
            self._psutil = None

    def sample(self) -> Sample:
        s = Sample()
        if not self.enabled:
            return s

        if self._nvml is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                s.gpu_util = float(util.gpu)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                s.gpu_mem_used = float(mem.used)
                s.gpu_mem_total = float(mem.total)
                s.gpu_temp = float(
                    self._nvml.nvmlDeviceGetTemperature(self._handle, 0)
                )
                s.gpu_power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                s.gpu_power_cap = (
                    self._nvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
                )
            except Exception:
                pass

        if self._torch is not None:
            try:
                s.torch_alloc = float(self._torch.cuda.memory_allocated(self.device_index))
                s.torch_reserved = float(
                    self._torch.cuda.memory_reserved(self.device_index)
                )
                if s.gpu_mem_total is None:
                    free, total = self._torch.cuda.mem_get_info(self.device_index)
                    s.gpu_mem_used = float(total - free)
                    s.gpu_mem_total = float(total)
            except Exception:
                pass

        if self._psutil is not None:
            try:
                s.cpu_percent = float(self._psutil.cpu_percent(interval=None))
                vm = self._psutil.virtual_memory()
                s.ram_used = float(vm.used)
                s.ram_total = float(vm.total)
            except Exception:
                pass

        return s

    def peak_vram(self) -> float | None:
        if self._torch is None:
            return None
        try:
            return float(self._torch.cuda.max_memory_reserved(self.device_index))
        except Exception:
            return None

    def reset_peak(self) -> None:
        if self._torch is not None:
            try:
                self._torch.cuda.reset_peak_memory_stats(self.device_index)
            except Exception:
                pass

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


class GpuGovernor:
    """Keeps a laptop GPU inside safe limits during long training runs.

    Three independent levers, because "don't use more than 90% GPU" can mean
    any of three things and they need different mechanisms:

    * **VRAM** — ``torch.cuda.set_per_process_memory_fraction`` caps this
      process at a fraction of total VRAM. Leaving headroom matters on a
      machine where the same card drives the display: starving the compositor
      is a well-known route to a driver reset.
    * **Utilisation** — there is no API to cap this, so it is duty-cycled: after
      each step, sleep a proportion of the step's own duration. A 90% target
      sleeps ``step_time * (1/0.9 - 1)`` ≈ 11% of the step time, which is also
      roughly the throughput you give up.
    * **Temperature** — a hard guard that blocks until the card cools. This is
      the one that actually prevents thermal shutdowns; utilisation capping
      only slows the approach to them.

    All limits are advisory and degrade to no-ops when NVML is unavailable.
    """

    def __init__(
        self,
        mon: "SysMon | None" = None,
        util_target: float | None = 90.0,
        temp_limit: float | None = 80.0,
        temp_resume: float | None = None,
        mem_fraction: float | None = 0.9,
        device_index: int = 0,
        verbose: bool = True,
    ):
        self.mon = mon
        self.util_target = util_target
        self.temp_limit = temp_limit
        self.temp_resume = temp_resume if temp_resume is not None else (
            (temp_limit - 6.0) if temp_limit else None
        )
        self.device_index = device_index
        self.verbose = verbose
        self.throttled_seconds = 0.0
        self.cooldowns = 0

        if mem_fraction:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(
                        float(mem_fraction), device_index
                    )
                    total = torch.cuda.get_device_properties(
                        device_index
                    ).total_memory / 1024**3
                    if verbose:
                        print(
                            f"[gpu] VRAM cap {mem_fraction:.0%} of {total:.1f} GB "
                            f"= {total * mem_fraction:.2f} GB for this process"
                        )
            except Exception as exc:
                if verbose:
                    print(f"[gpu] could not set VRAM cap: {exc}")

        if verbose:
            if util_target:
                print(f"[gpu] utilisation target {util_target:.0f}% (duty-cycled)")
            if temp_limit:
                print(
                    f"[gpu] thermal guard: pause above {temp_limit:.0f}C, "
                    f"resume below {self.temp_resume:.0f}C"
                )

    def after_step(self, step_seconds: float, on_wait=None) -> float:
        """Call once per training step. Returns seconds spent throttling."""
        slept = 0.0

        if self.util_target and 0 < self.util_target < 100 and step_seconds > 0:
            pause = step_seconds * (100.0 / self.util_target - 1.0)
            # Cap the per-step sleep so a pathological outlier step cannot
            # stall the run for minutes.
            pause = min(pause, max(step_seconds, 1.0))
            if pause > 0.001:
                time.sleep(pause)
                slept += pause

        if self.temp_limit and self.mon is not None:
            temp = self.mon.sample().gpu_temp
            if temp is not None and temp >= self.temp_limit:
                self.cooldowns += 1
                if self.verbose:
                    print(
                        f"\n[gpu] {temp:.0f}C >= {self.temp_limit:.0f}C — cooling "
                        f"until {self.temp_resume:.0f}C"
                    )
                t0 = time.time()
                while True:
                    if on_wait:
                        on_wait()
                    time.sleep(2.0)
                    t = self.mon.sample().gpu_temp
                    if t is None or t <= self.temp_resume:
                        break
                    if time.time() - t0 > 300:  # never hang forever
                        if self.verbose:
                            print("[gpu] cooldown exceeded 5 min — continuing anyway")
                        break
                waited = time.time() - t0
                slept += waited
                if self.verbose:
                    print(f"[gpu] resumed after {waited:.0f}s")

        self.throttled_seconds += slept
        return slept

    def summary(self) -> str:
        return (
            f"throttled {self.throttled_seconds:.0f}s total, "
            f"{self.cooldowns} thermal cooldown(s)"
        )
