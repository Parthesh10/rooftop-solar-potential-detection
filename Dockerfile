# Serves webapp/ only — the training stack (torch, segmentation-models-pytorch,
# ~2.5 GB) never enters this image. The app runs the exported ONNX graph, so
# onnxruntime (~50 MB) is all inference needs. See requirements-webapp.txt.
FROM python:3.13-slim

# curl: fetches the model from the public GitHub Release at build time (below).
# Nothing else compiles from source here, so no build-essential is needed —
# onnxruntime and opencv-python-headless both ship manylinux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-webapp.txt .
# opencv-python (with GUI bindings) drags in libGL, which this slim, headless
# image does not have and does not need. -headless is the same library, same
# version, minus the display code this server never calls — installed instead
# of opencv-python rather than uninstalled after, so libGL is never fetched.
RUN grep -v '^opencv-python==' requirements-webapp.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
    && pip install --no-cache-dir opencv-python-headless==5.0.0.93

COPY webapp/ webapp/

# The default checkpoint, fetched from the public GitHub Release rather than
# baked into the image or the repo — *.onnx is gitignored (F-26: exporting a
# second model into webapp/models/ must never silently become what is served,
# so the model directory ships empty and is filled explicitly, once, here).
# RSOLAR_MODEL_RELEASE lets a rebuild point at a newer release without editing
# this file; RSOLAR_MODEL_STEM must match the .onnx/.json filenames in it.
ARG RSOLAR_MODEL_RELEASE=v1.3-world25
ARG RSOLAR_MODEL_STEM=joint_v3_world25_20260908
# .dockerignore excludes webapp/models/ wholesale (research checkpoints and
# stale sidecars have no business in a serving image), which means Docker
# never copies the directory itself — an empty dir with nothing inside it to
# match. Create it explicitly or the curl below has nowhere to write.
RUN mkdir -p webapp/models
RUN curl -fSL -o webapp/models/${RSOLAR_MODEL_STEM}.onnx \
      "https://github.com/Parthesh10/rooftop-solar-potential-detection/releases/download/${RSOLAR_MODEL_RELEASE}/${RSOLAR_MODEL_STEM}.onnx" \
    && curl -fSL -o webapp/models/${RSOLAR_MODEL_STEM}.json \
      "https://github.com/Parthesh10/rooftop-solar-potential-detection/releases/download/${RSOLAR_MODEL_RELEASE}/${RSOLAR_MODEL_STEM}.json"

# Unprivileged: this process talks to the public internet (tile provider,
# PVGIS, Overpass) and has no reason to run as root.
RUN useradd -m -u 1000 rsolar && chown -R rsolar:rsolar /app
USER rsolar

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://127.0.0.1:8000/api/health || exit 1

# Platforms that inject $PORT (Render, Railway, Fly's default) override this;
# the shell form lets ${PORT:-8000} expand, which the exec form cannot do.
CMD uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8000}
