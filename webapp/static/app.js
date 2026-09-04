/* Rooftop Solar Potential — client.
   Plain ES2020, no build step: the whole app is one FastAPI process. */

'use strict';

const $ = (id) => document.getElementById(id);

const state = {
  cfg: null,
  map: null,
  aoi: null,          // {west, south, east, north}
  drawing: false,
  jobId: null,
  poll: null,
  result: null,
  region: null,       // /api/region-profile for the current map centre
  regionKey: null,
  touched: new Set(), // assumption fields the user has edited by hand
  calibrating: false,
};

/* ───────────────────────── boot ───────────────────────── */

async function boot() {
  try {
    state.cfg = await (await fetch('/api/config')).json();
  } catch {
    fatal('Cannot reach the server. Is `python -m webapp` still running?');
    return;
  }

  if (!state.cfg.tile_provider) {
    fatal(state.cfg.tile_provider_error || 'No imagery provider configured.');
    return;
  }
  if (!state.cfg.model_ready) {
    showError(
      (state.cfg.model_error || 'Model not loaded.') +
      '  Run:  python scripts/export_onnx.py'
    );
  }

  initMap();
  wireControls();
  initOnboarding();

  const p = state.cfg.tile_provider;
  $('attribution').innerHTML = p.attribution;

  // Default the threshold to whatever the model's own manifest says is optimal,
  // rather than a number baked into this file.
  try {
    state.model = await (await fetch('/api/model')).json();
    if (state.model.threshold) {
      $('threshold').value = state.model.threshold;
      $('threshold-out').textContent = (+state.model.threshold).toFixed(2);
    }
    if (state.model.tta) $('tta').checked = true;
  } catch { /* the model card is a nicety here, not a requirement */ }
}

function fatal(msg) {
  document.body.innerHTML =
    `<div style="display:grid;place-items:center;height:100vh;padding:40px;
                 font-family:var(--sans);color:#e6edf3;text-align:center">
       <div><h1 style="font-size:17px;margin:0 0 10px">Something is wrong</h1>
       <p style="color:#9aa7b4;font-size:13px;max-width:460px">${esc(msg)}</p></div>
     </div>`;
}

/* ───────────────────────── map ───────────────────────── */

function initMap() {
  const p = state.cfg.tile_provider;

  state.map = new maplibregl.Map({
    container: 'map',
    style: {
      version: 8,
      sources: {
        sat: {
          type: 'raster',
          tiles: [p.url],
          tileSize: 256,
          maxzoom: p.max_zoom,
          attribution: p.attribution,
        },
      },
      layers: [{ id: 'sat', type: 'raster', source: 'sat' }],
    },
    center: [77.4030, 23.2144],   // MANIT Bhopal — where the project started
    zoom: 16,
    maxZoom: p.max_zoom,
    attributionControl: { compact: true },
  });

  state.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
  state.map.addControl(new maplibregl.ScaleControl({ maxWidth: 130, unit: 'metric' }), 'bottom-right');
  state.map.addControl(
    new maplibregl.GeolocateControl({ positionOptions: { enableHighAccuracy: true } }),
    'top-right'
  );

  state.map.on('load', () => {
    // AOI rectangle
    state.map.addSource('aoi', { type: 'geojson', data: emptyFC() });
    state.map.addLayer({
      id: 'aoi-fill', type: 'fill', source: 'aoi',
      paint: { 'fill-color': '#f5a623', 'fill-opacity': 0.08 },
    });
    state.map.addLayer({
      id: 'aoi-line', type: 'line', source: 'aoi',
      paint: { 'line-color': '#f5a623', 'line-width': 2, 'line-dasharray': [2, 1.5] },
    });

    // Detected roofs
    state.map.addSource('roofs', { type: 'geojson', data: emptyFC() });
    state.map.addLayer({
      id: 'roofs-fill', type: 'fill', source: 'roofs',
      paint: {
        'fill-color': [
          'interpolate', ['linear'], ['get', 'confidence'],
          0.5, '#c2410c', 0.75, '#f5a623', 0.95, '#ffd97a',
        ],
        'fill-opacity': 0.55,
      },
    });
    state.map.addLayer({
      id: 'roofs-line', type: 'line', source: 'roofs',
      paint: { 'line-color': '#ffe9b0', 'line-width': 1, 'line-opacity': 0.85 },
    });
    state.map.addLayer({
      id: 'roofs-hl', type: 'line', source: 'roofs',
      filter: ['==', ['get', 'id'], -1],
      paint: { 'line-color': '#ffffff', 'line-width': 3 },
    });

    state.map.on('click', 'roofs-fill', (e) => {
      const f = e.features[0];
      if (!f) return;
      const pr = f.properties;
      new maplibregl.Popup({ closeButton: false, className: 'roof-popup' })
        .setLngLat(e.lngLat)
        .setHTML(
          `<div style="font-family:var(--mono,monospace);font-size:11px;line-height:1.7">
             <b>Roof #${pr.id}</b><br>
             area &nbsp;${(+pr.roof_area_m2).toLocaleString()} m²<br>
             usable ${(+pr.usable_area_m2).toLocaleString()} m²<br>
             conf. &nbsp;${(+pr.confidence).toFixed(2)}
           </div>`
        )
        .addTo(state.map);
      highlightRoof(+pr.id, true);
    });
    state.map.on('mouseenter', 'roofs-fill', () => { state.map.getCanvas().style.cursor = 'pointer'; });
    state.map.on('mouseleave', 'roofs-fill', () => { state.map.getCanvas().style.cursor = ''; });
  });

  setupDrawing();
}

const emptyFC = () => ({ type: 'FeatureCollection', features: [] });

function bboxFeature(b) {
  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature', properties: {},
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [b.west, b.south], [b.east, b.south],
          [b.east, b.north], [b.west, b.north], [b.west, b.south],
        ]],
      },
    }],
  };
}

/* ───────────────────────── drawing ───────────────────────── */

function setupDrawing() {
  const canvas = state.map.getCanvasContainer();
  let start = null;

  const toLngLat = (e) => {
    const r = canvas.getBoundingClientRect();
    const pt = new maplibregl.Point(e.clientX - r.left, e.clientY - r.top);
    return state.map.unproject(pt);
  };

  const onDown = (e) => {
    if (!state.drawing || e.button !== 0) return;
    e.preventDefault();
    start = toLngLat(e);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp, { once: true });
  };

  const onMove = (e) => {
    if (!start) return;
    const b = boundsFrom(start, toLngLat(e));
    setAOI(b, false);
    showDrawTip(b, e.clientX, e.clientY);
  };

  const onUp = (e) => {
    document.removeEventListener('mousemove', onMove);
    hideDrawTip();
    if (!start) return;
    const b = boundsFrom(start, toLngLat(e));
    start = null;
    stopDrawing();
    // Ignore an accidental click with no drag.
    if (Math.abs(b.east - b.west) < 1e-6 || Math.abs(b.north - b.south) < 1e-6) {
      setAOI(null);
      return;
    }
    setAOI(b, true);
  };

  canvas.addEventListener('mousedown', onDown);

  // Touch: a single-finger drag draws while in draw mode.
  canvas.addEventListener('touchstart', (e) => {
    if (!state.drawing || e.touches.length !== 1) return;
    e.preventDefault();
    const t = e.touches[0];
    start = toLngLat(t);
    const move = (ev) => {
      if (ev.touches.length !== 1 || !start) return;
      setAOI(boundsFrom(start, toLngLat(ev.touches[0])), false);
    };
    const end = () => {
      canvas.removeEventListener('touchmove', move);
      if (state.aoi) { stopDrawing(); setAOI(state.aoi, true); }
      start = null;
    };
    canvas.addEventListener('touchmove', move, { passive: false });
    canvas.addEventListener('touchend', end, { once: true });
  }, { passive: false });
}

function boundsFrom(a, b) {
  return {
    west: Math.min(a.lng, b.lng), east: Math.max(a.lng, b.lng),
    south: Math.min(a.lat, b.lat), north: Math.max(a.lat, b.lat),
  };
}

function startDrawing() {
  state.drawing = true;
  state.map.dragPan.disable();
  state.map.doubleClickZoom.disable();
  $('map').classList.add('drawing');
  $('draw-btn').classList.add('active');
  $('draw-label').textContent = 'Cancel';
  badge('Drag a box over the rooftops you want to analyse — Esc to cancel');
}

function stopDrawing() {
  state.drawing = false;
  state.map.dragPan.enable();
  state.map.doubleClickZoom.enable();
  $('map').classList.remove('drawing');
  $('draw-btn').classList.remove('active');
  $('draw-label').textContent = state.aoi ? 'Redraw area' : 'Draw area';
  badge(null);
  hideDrawTip();
}

/** Live w × h read-out pinned to the cursor while the box is being dragged. */
function showDrawTip(b, clientX, clientY) {
  const el = $('draw-tip');
  const midLat = (b.north + b.south) / 2;
  const w = haversine(b.west, midLat, b.east, midLat);
  const h = haversine(b.west, b.south, b.west, b.north);
  const tiles = estimateTiles(b);
  const over = tiles > state.cfg.max_tiles;

  el.innerHTML =
    `<b>${Math.round(w)} × ${Math.round(h)} m</b>
     <span class="${over ? 'dt-over' : ''}">${tiles} tile${tiles === 1 ? '' : 's'}${over ? ' — too big' : ''}</span>`;
  el.classList.toggle('over', over);
  el.style.left = (clientX + 16) + 'px';
  el.style.top = (clientY + 16) + 'px';
  el.hidden = false;
}

function hideDrawTip() { $('draw-tip').hidden = true; }

function setAOI(b, final) {
  state.aoi = b;
  state.map.getSource('aoi').setData(b ? bboxFeature(b) : emptyFC());
  $('clear-btn').disabled = !b;

  const readout = $('aoi-readout');
  if (!b) {
    readout.hidden = true;
    $('run-btn').disabled = true;
    return;
  }

  const midLat = (b.north + b.south) / 2;
  const w = haversine(b.west, midLat, b.east, midLat);
  const h = haversine(b.west, b.south, b.west, b.north);
  const areaKm2 = (w * h) / 1e6;
  const tiles = estimateTiles(b);
  const over = tiles > state.cfg.max_tiles;

  readout.hidden = false;
  readout.innerHTML =
    `<div class="r-row"><span>size</span><b>${Math.round(w)} × ${Math.round(h)} m</b></div>
     <div class="r-row"><span>area</span><b>${areaKm2.toFixed(3)} km²</b></div>
     <div class="r-row"><span>tiles @ z${state.cfg.serving_zoom}</span>
       <b class="${over ? 'over' : ''}">${tiles}${over ? ` / ${state.cfg.max_tiles} max` : ''}</b></div>`;

  $('run-btn').disabled = over || !state.cfg.model_ready;
  $('calibrate-btn').disabled = over || state.calibrating
                                || !state.cfg.model_ready;
  if (over) {
    showError(`That area needs ${tiles} imagery tiles; the limit is ` +
              `${state.cfg.max_tiles}. Draw a smaller box.`);
  } else {
    hideError();
  }

  if (final) {
    state.map.fitBounds([[b.west, b.south], [b.east, b.north]],
                        { padding: 70, maxZoom: state.cfg.serving_zoom, duration: 550 });
  }
}

function estimateTiles(b) {
  const z = state.cfg.serving_zoom;
  const n = 2 ** z;
  const tx = (lon) => Math.floor((lon + 180) / 360 * n);
  const ty = (lat) => {
    const r = lat * Math.PI / 180;
    return Math.floor((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n);
  };
  return (tx(b.east) - tx(b.west) + 1) * (ty(b.south) - ty(b.north) + 1);
}

function haversine(lon1, lat1, lon2, lat2) {
  const R = 6371008.8, rad = Math.PI / 180;
  const p1 = lat1 * rad, p2 = lat2 * rad;
  const dp = p2 - p1, dl = (lon2 - lon1) * rad;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function badge(text) {
  const el = $('map-badge');
  if (!text) { el.hidden = true; return; }
  el.textContent = text;
  el.hidden = false;
}

/* ───────────────────────── controls ───────────────────────── */

function wireControls() {
  $('draw-btn').onclick = () => (state.drawing ? stopDrawing() : startDrawing());

  $('clear-btn').onclick = () => {
    setAOI(null);
    $('calibrate-btn').disabled = true;
    state.map.getSource('roofs').setData(emptyFC());
    $('results').hidden = true;
    $('layer-toggle').hidden = true;
    $('draw-label').textContent = 'Draw area';
    state.result = null;
    hideError();
  };

  // Live slider read-outs
  const bind = (id, fmt) => {
    const el = $(id), out = $(id + '-out');
    const upd = () => { out.textContent = fmt(+el.value); };
    el.addEventListener('input', upd); upd();
  };
  bind('packing', (v) => v.toFixed(2));
  bind('efficiency', (v) => Math.round(v * 100) + '%');
  bind('losses', (v) => v + '%');
  bind('threshold', (v) => v.toFixed(2));

  $('threshold').addEventListener('input', () => {
    state.thresholdTouched = true;
    $('threshold-auto').hidden = true;
  });

  // A regional default must never overwrite a number the user typed.
  Object.keys(REGION_FIELDS).forEach((id) => {
    $(id).addEventListener('change', () => state.touched.add(id));
  });

  $('calibrate-btn').onclick = runCalibration;
  state.map.on('moveend', () => scheduleRegionRefresh());
  scheduleRegionRefresh(0);

  $('reset-assumptions').onclick = async () => {
    const a = await (await fetch('/api/assumptions')).json();
    const d = a.defaults;
    $('packing').value = d.packing_factor;
    $('efficiency').value = d.module_efficiency;
    $('losses').value = d.system_losses_pct;
    $('threshold').value = state.model?.threshold ?? 0.5;
    state.thresholdTouched = false;
    $('threshold-auto').hidden = false;
    $('tta').checked = false;
    $('tariff').value = d.tariff_per_kwh;
    $('cost').value = d.cost_per_kwp;
    $('subsidy').value = d.subsidy;
    $('emission').value = d.grid_emission_kg_per_kwh;
    $('currency-symbol').value = d.currency_symbol;
    $('tilt').value = '';
    ['packing', 'efficiency', 'losses', 'threshold']
      .forEach((i) => $(i).dispatchEvent(new Event('input')));
    // Reset means "back to what this place implies", not "back to India".
    state.touched.clear();
    applyRegionProfile();
  };

  $('run-btn').onclick = runAnalysis;
  $('results-close').onclick = () => { $('results').hidden = true; };

  $('show-roofs').onchange = (e) => {
    const v = e.target.checked ? 'visible' : 'none';
    ['roofs-fill', 'roofs-line', 'roofs-hl']
      .forEach((l) => state.map.setLayoutProperty(l, 'visibility', v));
  };
  $('roof-opacity').oninput = (e) => {
    state.map.setPaintProperty('roofs-fill', 'fill-opacity', +e.target.value / 100);
  };

  wireSearch();

  $('about-btn').onclick = showAbout;
  $('about-close').onclick = () => { $('about-modal').hidden = true; };
  $('about-modal').addEventListener('click', (e) => {
    if (e.target.id === 'about-modal') $('about-modal').hidden = true;
  });
  document.addEventListener('keydown', (e) => {
    // Keyboard shortcuts, but never while the user is typing in a field.
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);

    if (e.key === 'Escape') {
      $('about-modal').hidden = true;
      if (state.drawing) stopDrawing();
      return;
    }
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;

    if (e.key === 'd' || e.key === 'D') {
      e.preventDefault();
      state.drawing ? stopDrawing() : startDrawing();
    } else if (e.key === 'Enter' && !$('run-btn').disabled) {
      e.preventDefault();
      runAnalysis();
    } else if (e.key === '/') {
      e.preventDefault();
      $('search-input').focus();
      $('search-input').select();
    }
  });

  $('export-geojson').onclick = exportGeoJSON;
  $('export-csv').onclick = exportCSV;
}

/* ───────────────────────── search ─────────────────────────
   Type-ahead over Photon (Komoot's geocoder, built for autocomplete) with
   Nominatim as the fallback. Also accepts raw coordinates, which is the fastest
   path when someone already knows exactly where they want to look. */

const search = {
  items: [], active: -1, seq: 0, timer: null, cache: new Map(),
};

const RECENT_KEY = 'rsolar.recent';

function loadRecent() {
  try { return JSON.parse(localStorage.getItem(RECENT_KEY)) || []; }
  catch { return []; }
}

function saveRecent(item) {
  try {
    const list = loadRecent().filter((r) => r.label !== item.label);
    list.unshift(item);
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 5)));
  } catch { /* private mode: recents are a convenience, not a requirement */ }
}

/** "23.21, 77.40" / "23.21 77.40" / "23.21N 77.40E" -> a result, or null. */
function parseCoords(q) {
  const m = q.trim().match(
    /^(-?\d+(?:\.\d+)?)\s*°?\s*([NnSs])?\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*°?\s*([EeWw])?$/);
  if (!m) return null;
  let lat = parseFloat(m[1]);
  let lon = parseFloat(m[3]);
  if (m[2] && m[2].toLowerCase() === 's') lat = -lat;
  if (m[4] && m[4].toLowerCase() === 'w') lon = -lon;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return {
    label: `${lat.toFixed(5)}, ${lon.toFixed(5)}`,
    detail: 'coordinates', lat, lon, kind: 'coords',
  };
}

function normalisePhoton(json) {
  return (json.features || []).map((f) => {
    const p = f.properties || {};
    const [lon, lat] = f.geometry.coordinates;
    const label = p.name
      || [p.street, p.housenumber].filter(Boolean).join(' ')
      || p.city || p.county || 'Unnamed';
    const detail = [p.street && p.name !== p.street ? p.street : null,
                    p.district, p.city, p.county, p.state, p.country]
      .filter(Boolean).filter((v, i, a) => a.indexOf(v) === i).join(', ');
    return { label, detail, lat, lon, kind: p.osm_value || p.type || 'place' };
  });
}

function normaliseNominatim(json) {
  return (json || []).map((r) => {
    const parts = r.display_name.split(',').map((s) => s.trim());
    return {
      label: parts[0], detail: parts.slice(1).join(', '),
      lat: +r.lat, lon: +r.lon, kind: r.type || 'place',
    };
  });
}

async function fetchSuggestions(q) {
  if (search.cache.has(q)) return search.cache.get(q);

  let out = [];
  try {
    const url = `https://photon.komoot.io/api/?limit=6&q=${encodeURIComponent(q)}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error('photon ' + r.status);
    out = normalisePhoton(await r.json());
  } catch {
    try {
      const url = 'https://nominatim.openstreetmap.org/search?format=json&limit=6&q='
                + encodeURIComponent(q);
      const r = await fetch(url, { headers: { Accept: 'application/json' } });
      out = normaliseNominatim(await r.json());
    } catch {
      return null;   // both geocoders unreachable
    }
  }
  search.cache.set(q, out);
  return out;
}

const KIND_ICON = {
  coords: '⌖', city: '🏙', town: '🏙', village: '🏘', suburb: '🏘',
  house: '🏠', residential: '🏠', building: '🏢', university: '🎓',
  school: '🎓', hospital: '🏥', industrial: '🏭', recent: '🕘',
};
const kindIcon = (k) => KIND_ICON[k] || '📍';

function renderSuggestions(items, { emptyMessage = null, heading = null } = {}) {
  const box = $('search-results');
  search.items = items;
  search.active = -1;

  if (!items.length) {
    box.innerHTML = `<div class="sr-empty">${esc(emptyMessage || 'No matches')}</div>`;
    box.hidden = false;
    $('search-input').setAttribute('aria-expanded', 'true');
    return;
  }

  box.innerHTML =
    (heading ? `<div class="sr-heading">${esc(heading)}</div>` : '')
    + items.map((it, i) => `
      <div class="sr-item" role="option" data-i="${i}" id="sr-opt-${i}">
        <span class="sr-ico">${kindIcon(it.kind)}</span>
        <span class="sr-text">
          <span class="sr-label">${esc(it.label)}</span>
          ${it.detail ? `<span class="sr-detail">${esc(it.detail)}</span>` : ''}
        </span>
      </div>`).join('');

  box.querySelectorAll('.sr-item').forEach((el) => {
    const i = +el.dataset.i;
    el.onmouseenter = () => setActiveSuggestion(i, false);
    el.onmousedown = (e) => { e.preventDefault(); chooseSuggestion(i); };
  });

  box.hidden = false;
  $('search-input').setAttribute('aria-expanded', 'true');
}

function setActiveSuggestion(i, scroll = true) {
  const box = $('search-results');
  const els = [...box.querySelectorAll('.sr-item')];
  if (!els.length) return;
  search.active = (i + els.length) % els.length;
  els.forEach((el, n) => el.classList.toggle('active', n === search.active));
  $('search-input').setAttribute('aria-activedescendant', `sr-opt-${search.active}`);
  if (scroll) els[search.active].scrollIntoView({ block: 'nearest' });
}

function closeSuggestions() {
  $('search-results').hidden = true;
  $('search-input').setAttribute('aria-expanded', 'false');
  search.active = -1;
}

function chooseSuggestion(i) {
  const it = search.items[i];
  if (!it) return;
  closeSuggestions();
  $('search-input').value = it.label;
  $('search-clear').hidden = false;
  saveRecent(it);
  goTo(it);
}

function goTo(it) {
  // z-1 frames a block or two: enough context to pick out one building without
  // making the user zoom out again.
  state.map.flyTo({ center: [it.lon, it.lat],
                    zoom: state.cfg.serving_zoom - 1, duration: 1300 });
  state.map.once('moveend', () => {
    if (!state.aoi) badge('Now click "Draw area" and drag a box over the roofs');
    setTimeout(() => badge(null), 5200);
  });
}

function showRecentOrHint() {
  const recent = loadRecent();
  if (recent.length) {
    renderSuggestions(recent.map((r) => ({ ...r, kind: 'recent' })),
                      { heading: 'Recent' });
  } else {
    closeSuggestions();
  }
}

async function runSearch(q) {
  const seq = ++search.seq;

  const coords = parseCoords(q);
  if (coords) { renderSuggestions([coords], { heading: 'Go to coordinates' }); return; }

  if (q.length < 2) { showRecentOrHint(); return; }

  $('search-spin').hidden = false;
  const items = await fetchSuggestions(q);
  if (seq !== search.seq) return;         // a newer keystroke already won
  $('search-spin').hidden = true;

  if (items === null) {
    renderSuggestions([], {
      emptyMessage: 'Search is unavailable — pan the map, or paste "lat, lon".' });
  } else {
    renderSuggestions(items, {
      emptyMessage: `No match for "${q}". Try adding a city or country.` });
  }
}

function wireSearch() {
  const input = $('search-input');

  input.addEventListener('input', () => {
    const q = input.value.trim();
    $('search-clear').hidden = !q;
    clearTimeout(search.timer);
    if (!q) { $('search-spin').hidden = true; showRecentOrHint(); return; }
    // 220 ms: long enough not to fire on every keystroke, short enough that the
    // list feels like it is keeping up.
    search.timer = setTimeout(() => runSearch(q), 220);
  });

  input.addEventListener('focus', () => {
    if (!input.value.trim()) showRecentOrHint();
    else if (search.items.length) $('search-results').hidden = false;
  });

  input.addEventListener('keydown', (e) => {
    const open = !$('search-results').hidden;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!open) { runSearch(input.value.trim()); return; }
      setActiveSuggestion(search.active + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveSuggestion(search.active - 1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      clearTimeout(search.timer);
      if (open && search.active >= 0) chooseSuggestion(search.active);
      else if (open && search.items.length) chooseSuggestion(0);
      else runSearch(input.value.trim()).then(() => {
        if (search.items.length) chooseSuggestion(0);
      });
    } else if (e.key === 'Escape') {
      closeSuggestions();
      input.blur();
    }
  });

  $('search-clear').onclick = () => {
    input.value = '';
    $('search-clear').hidden = true;
    input.focus();
    showRecentOrHint();
  };

  document.addEventListener('mousedown', (e) => {
    if (!e.target.closest('.search-box')) closeSuggestions();
  });

  $('here-btn').onclick = () => {
    if (!navigator.geolocation) {
      showError('This browser will not share a location.');
      return;
    }
    badge('Finding your location…');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        badge(null);
        goTo({ lat: pos.coords.latitude, lon: pos.coords.longitude });
      },
      () => {
        badge(null);
        showError('Could not get your location. Browsers only allow this over '
                + 'HTTPS or on localhost, and it needs permission.');
      },
      { enableHighAccuracy: true, timeout: 10000 });
  };
}

/* ───────────────────────── analysis ───────────────────────── */


/* ─────────────── region profile & detection calibration ─────────────── */

/* Fields the region table can pre-fill, and where they live in the DOM. Once
   the user edits one it is never overwritten again — a regional default is a
   starting point, not an opinion about their electricity bill. */
const REGION_FIELDS = {
  packing: 'packing_factor',
  tariff: 'tariff_per_kwh',
  cost: 'cost_per_kwp',
  emission: 'grid_emission_kg_per_kwh',
  'currency-symbol': 'currency_symbol',
};

let regionTimer = null;

function scheduleRegionRefresh(delay = 400) {
  clearTimeout(regionTimer);
  regionTimer = setTimeout(refreshRegionProfile, delay);
}

async function refreshRegionProfile() {
  if (!state.map) return;
  const c = state.map.getCenter();
  // ~1 km of granularity: panning across a city must not re-ask every frame.
  const key = `${c.lat.toFixed(2)},${c.lng.toFixed(2)}`;
  if (key === state.regionKey) return;
  state.regionKey = key;
  try {
    const r = await fetch(`/api/region-profile?lat=${c.lat}&lon=${c.lng}`);
    if (!r.ok) return;
    state.region = await r.json();
    applyRegionProfile();
  } catch { /* the app is fully usable without this */ }
}

function applyRegionProfile() {
  const rp = state.region;
  if (!rp) return;
  const eco = rp.region.economics;

  for (const [id, key] of Object.entries(REGION_FIELDS)) {
    if (state.touched.has(id)) continue;
    const el = $(id);
    if (!el || eco[key] === undefined) continue;
    el.value = eco[key];
    el.dispatchEvent(new Event('input'));
  }

  if (!state.thresholdTouched) {
    const t = rp.calibration.threshold;
    $('threshold').value = t;
    $('threshold-out').textContent = (+t).toFixed(2);
    const tag = $('threshold-auto');
    tag.hidden = false;
    tag.textContent = rp.calibration.source === 'reference' ? 'measured' : 'auto';
    tag.title = rp.calibration.steps.join('  →  ');
  }

  const conf = rp.region.economics_confidence;
  const measured = rp.calibration.source === 'reference';
  const note = conf === 'none'
    ? 'no local rates on file — the costs below are placeholders'
    : `indicative ${conf}-confidence rates, in ${eco.currency}`;
  $('region-chip').hidden = false;
  $('region-chip').innerHTML =
    `<b>${esc(rp.region.name)}</b>
     <span class="rc-note">${esc(note)}</span>` +
    (measured
      ? '<span class="rc-note">· detection measured here</span>'
      : '');

  $('calibrate-btn').disabled = !state.aoi || state.calibrating
                               || !state.cfg.model_ready;
}

/* Poll any job to a terminal state. `pollJob` stays as it is for analysis; this
   is the generic version calibration uses. */
function awaitJob(jid, onTick) {
  return new Promise((resolve, reject) => {
    const t = setInterval(async () => {
      try {
        const j = await (await fetch(`/api/jobs/${jid}`)).json();
        onTick?.(j);
        if (j.state === 'done') { clearInterval(t); resolve(j.result); }
        else if (j.state === 'error') { clearInterval(t); reject(new Error(j.error)); }
      } catch (e) { clearInterval(t); reject(e); }
    }, 700);
  });
}

async function runCalibration() {
  if (!state.aoi || state.calibrating) return;
  state.calibrating = true;
  $('calibrate-btn').disabled = true;
  $('calib-report').hidden = true;
  $('calib-state').textContent = 'starting…';

  try {
    const r = await fetch('/api/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bounds: state.aoi, zoom: state.cfg.serving_zoom }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
    const res = await awaitJob((await r.json()).job_id,
                               (j) => { $('calib-state').textContent = j.message; });

    $('calib-state').textContent = '';
    renderCalibrationReport(res);

    // A measurement beats the slider's current guess, so adopt it — and drop
    // the "touched" flag, because this value is no longer a guess.
    const t = res.calibration.threshold;
    $('threshold').value = t;
    $('threshold-out').textContent = (+t).toFixed(2);
    state.thresholdTouched = false;
    $('threshold-auto').hidden = false;
    $('threshold-auto').textContent =
      res.calibration.source === 'reference' ? 'measured' : 'auto';
    state.regionKey = null;          // force a re-read so the chip catches up
    scheduleRegionRefresh(0);
  } catch (e) {
    $('calib-state').textContent = '';
    const box = $('calib-report');
    box.hidden = false;
    box.className = 'calib-report bad';
    box.textContent = e.message;
  } finally {
    state.calibrating = false;
    $('calibrate-btn').disabled = !state.aoi;
  }
}

function renderCalibrationReport(res) {
  const cal = res.calibration, ref = res.reference || {};
  const box = $('calib-report');
  box.hidden = false;
  box.className = 'calib-report' + (cal.verdict === 'needs_finetuning' ? ' bad' : '');
  box.innerHTML =
    `<div class="kv">
       <div class="kv-row"><span>threshold</span><span>${cal.threshold}</span></div>
       <div class="kv-row"><span>mapped buildings used</span>
         <span>${ref.usable ?? '—'} of ${ref.found ?? '—'}</span></div>
       <div class="kv-row"><span>of those, found</span>
         <span>${cal.diagnostics?.reference_recall != null
                 ? Math.round(cal.diagnostics.reference_recall * 100) + '%' : '—'}</span></div>
       <div class="kv-row"><span>model is silent on</span>
         <span>${cal.diagnostics?.silent_fraction != null
                 ? Math.round(cal.diagnostics.silent_fraction * 100) + '% of them'
                 : '—'}</span></div>
     </div>
     <p class="hint tiny" style="margin-top:8px">${esc(cal.note)}</p>` +
    renderRecallBySize(cal.diagnostics?.recall_by_size) +
    `<p class="hint tiny">${esc(ref.caveat || '')}</p>`;
}

/* Which of the user's buildings the number can be trusted for. A flat 30%
   recall and a 30% that is 8% on houses and 46% on large roofs are completely
   different answers, and only the second one is actionable. */
function renderRecallBySize(bands) {
  if (!bands || !bands.length) return '';
  const rows = bands.map((b) => {
    const label = b.to_m2 == null ? `${b.from_m2}+ m²`
                                  : `${b.from_m2}–${b.to_m2} m²`;
    return `<div class="kv-row"><span>${label} (n=${b.n})</span>
              <span>${Math.round(b.recall * 100)}%</span></div>`;
  }).join('');
  return `<p class="hint tiny" style="margin-top:10px">Found, by roof size:</p>
          <div class="kv">${rows}</div>`;
}

function renderCalibration(res) {
  const cal = res.calibration;
  const block = $('detection-block');
  if (!cal) { block.hidden = true; return; }
  block.hidden = false;

  const labels = {
    reference: 'measured here',
    histogram: 'self-calibrated',
    prior: 'regional prior',
    user: 'set by hand',
  };
  const pill = $('calib-pill');
  pill.textContent = labels[cal.source] || cal.source;
  pill.className = 'pill calib-' + cal.source;

  const det = res.detection || {};
  const reg = res.region;
  const rows = [
    ['threshold used', det.threshold ?? cal.threshold],
    ['chosen by', labels[cal.source] || cal.source],
    ['allowed range here', `${cal.band[0]} – ${cal.band[1]}`],
    ['region', reg ? reg.name : '—'],
    ['closing kernel', det.morph_kernel_px ? `${det.morph_kernel_px} px` : '—'],
  ];
  const morph = cal.diagnostics?.morphology;
  if (morph?.median_component_m2 != null) {
    rows.push(['median roof detected', `${morph.median_component_m2} m²`]);
  }

  $('calib-detail').innerHTML =
    `<div class="kv">${rows.map(([k, v]) =>
       `<div class="kv-row"><span>${esc(k)}</span><span>${esc(String(v))}</span></div>`
     ).join('')}</div>
     <ol class="calib-steps">${(cal.steps || [])
       .map((x) => `<li>${esc(x)}</li>`).join('')}</ol>` +
    (morph?.reason ? `<p class="hint tiny">${esc(morph.reason)}.</p>` : '') +
    (cal.verdict === 'needs_finetuning'
      ? `<p class="calib-warn">${esc(cal.note)}</p>` : '');
}

function payload() {
  const tilt = $('tilt').value.trim();
  return {
    bounds: state.aoi,
    zoom: state.cfg.serving_zoom,
    // null = let the server choose based on how well the model knows this
    // region. Only override once the user has actually moved the slider.
    threshold: state.thresholdTouched ? +$('threshold').value : null,
    use_calibration: true,
    tta: $('tta').checked,
    packing_factor: +$('packing').value,
    module_efficiency: +$('efficiency').value,
    system_losses_pct: +$('losses').value,
    tilt_deg: tilt === '' ? null : +tilt,
    tariff_per_kwh: +$('tariff').value || 0,
    cost_per_kwp: +$('cost').value || 0,
    subsidy: +$('subsidy').value || 0,
    grid_emission_kg_per_kwh: +$('emission').value || 0,
    currency_symbol: $('currency-symbol').value || '',
  };
}

async function runAnalysis() {
  if (!state.aoi) return;
  hideError();
  dismissOnboarding();
  $('run-btn').disabled = true;
  $('run-label').textContent = 'Working…';
  $('progress-wrap').hidden = false;
  setProgress(0.02, 'starting…', 'fetching');

  try {
    const r = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload()),
    });
    if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
    state.jobId = (await r.json()).job_id;
    pollJob();
  } catch (e) {
    finishRun();
    showError(e.message);
  }
}

function pollJob() {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    try {
      const j = await (await fetch(`/api/jobs/${state.jobId}`)).json();
      setProgress(j.progress, j.message, j.state);
      if (j.state === 'done') {
        clearInterval(state.poll);
        finishRun();
        renderResult(j.result);
      } else if (j.state === 'error') {
        clearInterval(state.poll);
        finishRun();
        showError(j.error);
      }
    } catch {
      clearInterval(state.poll);
      finishRun();
      showError('Lost contact with the server.');
    }
  }, 650);
}

function finishRun() {
  $('run-btn').disabled = false;
  $('run-label').textContent = 'Detect rooftops';
  setTimeout(() => { $('progress-wrap').hidden = true; }, 900);
}

const ONBOARD_KEY = 'rsolar.onboarded';

function dismissOnboarding() {
  $('onboard').classList.add('gone');
  try { localStorage.setItem(ONBOARD_KEY, '1'); } catch { /* private mode */ }
}

function initOnboarding() {
  let seen = false;
  try { seen = localStorage.getItem(ONBOARD_KEY) === '1'; } catch { /* ignore */ }
  if (seen) $('onboard').classList.add('gone');
  $('onboard-close').onclick = dismissOnboarding;
}

function setProgress(p, msg, stage) {
  $('progress-bar').style.width = Math.max(2, p * 100) + '%';
  $('progress-msg').textContent = msg || '';
  if (!stage) return;
  const order = ['fetching', 'detecting', 'measuring'];
  const at = order.indexOf(stage);
  document.querySelectorAll('.progress-steps span').forEach((el) => {
    const i = order.indexOf(el.dataset.stage);
    el.classList.toggle('done', at > i);
    el.classList.toggle('active', at === i);
  });
}

function showError(msg) {
  const b = $('error-box');
  b.textContent = msg;
  b.hidden = false;
}
function hideError() { $('error-box').hidden = true; }

/* ───────────────────────── results ───────────────────────── */

function renderResult(res) {
  state.result = res;
  const s = res.summary;
  const sym = s.currency_symbol || '';

  state.map.getSource('roofs').setData(res.geojson);
  $('layer-toggle').hidden = false;
  $('lt-count').textContent =
    `${res.geojson.features.length} roof${res.geojson.features.length === 1 ? '' : 's'}`;
  $('results').hidden = false;
  document.querySelector('.results-scroll').scrollTop = 0;

  // Geographic coverage: say plainly whether the model has seen anything like
  // this region, before the user reads a confident-looking number.
  const cov = res.coverage;
  const covHtml = cov ? `
    <div class="coverage cov-${cov.level}">
      <div class="cov-head">
        <span class="cov-dot"></span>
        ${cov.level === 'trained' ? 'Inside the training region'
          : cov.level === 'regional' ? 'Near the training region'
          : 'Outside the training region'}
      </div>
      <p>${esc(cov.note)}</p>
    </div>` : '';

  $('warnings').innerHTML = covHtml + (res.warnings || [])
    .map((w) => `<div class="warn-item">${esc(w)}</div>`).join('');

  // Cards
  const cards = [
    ['hero wide', 'Annual generation', fmtInt(s.annual_kwh), 'kWh / year',
      `${s.capacity_kwp} kWp system · ${s.specific_yield_kwh_per_kwp} kWh per kWp here`],
    ['', 'Roof detected', fmtInt(s.roof_area_m2), 'm²',
      `${s.building_count} building${s.building_count === 1 ? '' : 's'}`],
    ['', 'Usable for panels', fmtInt(s.usable_area_m2), 'm²',
      `packing factor ${s.packing_factor}`],
    ['', 'Annual savings', sym + fmtInt(s.annual_savings), '',
      `at ${sym}${res.assumptions.tariff_per_kwh}/kWh`],
    ['', 'Payback', s.payback_years ?? '—', s.payback_years ? 'years' : '',
      s.net_cost ? `net cost ${sym}${fmtInt(s.net_cost)}` : 'set a cost per kWp'],
    ['', 'CO₂ avoided', fmtInt(s.co2_avoided_kg_per_year), 'kg / year',
      `≈ ${fmtInt(s.trees_equivalent)} trees`],
    ['', '25-year savings', sym + fmtInt(s.lifetime_savings), '',
      `${s.co2_avoided_t_over_lifetime} t CO₂ avoided`],
  ];
  $('summary-cards').innerHTML = cards.map(([cls, label, val, unit, sub]) => `
    <div class="card ${cls}">
      <div class="card-label">${label}</div>
      <div class="card-value">${val}<span class="card-unit">${unit}</span></div>
      <div class="card-sub">${esc(sub)}</div>
    </div>`).join('');

  renderCalibration(res);
  renderSolarResource(res.solar_resource, s);

  drawChart(s.monthly_kwh);
  $('chart-note').textContent =
    `${s.irradiance_source}. Tilt ${s.optimal_tilt_deg}°, azimuth ${s.azimuth_deg}°.`
    + (s.irradiance_ok ? '' : '  ⚠ These are fallback numbers, not site data.');

  // Calculation chain
  const chain = [
    ['detected roof footprint', '', fmtInt(s.roof_area_m2) + ' m²'],
    ['packing factor', '×', s.packing_factor],
    ['usable PV area', '=', fmtInt(s.usable_area_m2) + ' m²'],
    ['module efficiency', '×', (s.module_efficiency ?? res.assumptions.module_efficiency)],
    ['capacity', '=', s.capacity_kwp + ' kWp'],
    ['site yield', '×', s.specific_yield_kwh_per_kwp + ' kWh/kWp'],
    ['annual generation', '=', fmtInt(s.annual_kwh) + ' kWh'],
  ];
  $('chain').innerHTML = chain.map(([label, op, val], i) => `
    <div class="chain-row ${i === chain.length - 1 ? 'total' : ''}">
      <span><span class="c-op">${op}</span> ${label}</span>
      <span class="c-val">${val}</span>
    </div>`).join('');

  // Table
  const feats = res.geojson.features;
  $('roof-count').textContent = feats.length;
  const rows = feats.slice(0, 300).map((f) => {
    const p = f.properties;
    return `<tr data-id="${p.id}">
      <td>${p.id}</td><td>${fmtInt(p.roof_area_m2)}</td>
      <td>${fmtInt(p.usable_area_m2)}</td><td>${(+p.confidence).toFixed(2)}</td></tr>`;
  }).join('');
  const tbody = document.querySelector('#roof-table tbody');
  tbody.innerHTML = rows || '<tr><td colspan="4">nothing detected</td></tr>';
  tbody.querySelectorAll('tr[data-id]').forEach((tr) => {
    const id = +tr.dataset.id;
    tr.onmouseenter = () => highlightRoof(id, false);
    tr.onmouseleave = () => highlightRoof(-1, false);
    tr.onclick = () => {
      const f = feats.find((x) => x.properties.id === id);
      if (f) state.map.fitBounds(ringBounds(f.geometry.coordinates[0]),
                                 { padding: 130, duration: 600 });
      highlightRoof(id, true);
    };
  });
  $('table-note').textContent = feats.length > 300
    ? `Showing the 300 largest of ${feats.length}. Export for the full list.`
    : `Sorted largest first. Confidence is the model's mean probability inside each roof.`;

  // Inputs used
  const a = res.assumptions, im = res.imagery;
  $('used-assumptions').innerHTML = [
    ['Packing factor', a.packing_factor],
    ['Module efficiency', (a.module_efficiency * 100).toFixed(1) + '%'],
    ['System losses', a.system_losses_pct + '%'],
    ['Tariff', sym + a.tariff_per_kwh + ' / kWh'],
    ['Cost per kWp', sym + fmtInt(a.cost_per_kwp)],
    ['Subsidy', sym + fmtInt(a.subsidy)],
    ['Grid CO₂', a.grid_emission_kg_per_kwh + ' kg/kWh'],
    ['Imagery', `${im.provider} · z${im.zoom}`],
    ['Ground resolution', im.metres_per_pixel + ' m/px'],
    ['Tiles analysed', `${im.tiles}${im.tiles_failed ? ` (${im.tiles_failed} failed)` : ''}`],
    ['Model', `${res.model.architecture} / ${res.model.encoder}`],
    ['Model val IoU', (res.model.metrics?.val?.iou ?? '—')],
    ['Detection threshold', `${res.detection.threshold}`
      + (res.detection.threshold_auto ? ' (auto for this region)' : ' (you set this)')],
    ['High accuracy (TTA)', res.detection.tta ? 'on' : 'off'],
    ['Mean confidence', s.mean_confidence],
    ['Roof coverage of AOI', s.roof_coverage_pct + '%'],
  ].map(([k, v]) => `<div class="kv-row"><span>${k}</span><span>${esc(String(v))}</span></div>`).join('');
}

function highlightRoof(id, persist) {
  state.map.setFilter('roofs-hl', ['==', ['get', 'id'], id]);
  if (persist) {
    document.querySelectorAll('#roof-table tbody tr').forEach((tr) => {
      tr.classList.toggle('hl', +tr.dataset.id === id);
    });
  }
}

function ringBounds(ring) {
  let w = 180, s = 90, e = -180, n = -90;
  ring.forEach(([lon, lat]) => {
    w = Math.min(w, lon); e = Math.max(e, lon);
    s = Math.min(s, lat); n = Math.max(n, lat);
  });
  return [[w, s], [e, n]];
}

/* ───────────────── solar resource (exposure) ─────────────────
   How much sun this *place* gets, independent of the roof or the system. It is
   what makes two locations comparable, and it is the physical input everything
   downstream multiplies. */

/** Rough global bands for annual GHI, so a number has a meaning attached. */
function ghiBand(ghi) {
  if (ghi == null) return null;
  if (ghi >= 2000) return { label: 'Exceptional', cls: 'b-exceptional' };
  if (ghi >= 1700) return { label: 'Very good',   cls: 'b-verygood' };
  if (ghi >= 1400) return { label: 'Good',        cls: 'b-good' };
  if (ghi >= 1100) return { label: 'Moderate',    cls: 'b-moderate' };
  return { label: 'Low', cls: 'b-low' };
}

function renderSolarResource(r, summary) {
  const block = $('resource-block');
  if (!r || !r.ok) {
    block.hidden = false;
    $('resource-scale').textContent = '';
    $('resource-cards').innerHTML = '';
    $('resource-chart').innerHTML = '';
    $('resource-legend').innerHTML = '';
    $('resource-note').textContent =
      'Solar exposure data is unavailable — PVGIS could not be reached. The '
      + 'generation estimate below is using fallback numbers.';
    return;
  }
  block.hidden = false;

  const band = ghiBand(r.annual_ghi_kwh_m2);
  $('resource-scale').textContent = band ? band.label : '';
  $('resource-scale').className = 'pill ' + (band ? band.cls : '');

  const cards = [
    ['Annual sunlight', fmtInt(r.annual_ghi_kwh_m2), 'kWh/m²',
     'on a flat surface'],
    ['Peak sun hours', r.peak_sun_hours_per_day, 'h/day',
     'full-strength-equivalent'],
    ['Sunniest', r.best_month.name, `${Math.round(r.best_month.value)} kWh/m²`,
     `${r.seasonality_ratio}× the lowest month`],
    ['Lowest', r.worst_month.name, `${Math.round(r.worst_month.value)} kWh/m²`,
     'plan storage around this'],
  ];
  $('resource-cards').innerHTML = cards.map(([l, v, u, s]) => `
    <div class="mini-card">
      <div class="mc-label">${l}</div>
      <div class="mc-value">${v}<span class="mc-unit">${esc(u)}</span></div>
      <div class="mc-sub">${esc(s)}</div>
    </div>`).join('');

  drawResourceChart(r, summary);

  $('resource-legend').innerHTML = `
    <span class="lg"><i class="lg-bar"></i>Sunlight on a flat surface (kWh/m²)</span>
    <span class="lg"><i class="lg-line"></i>On your panels at ${summary.optimal_tilt_deg}° tilt</span>
    <span class="lg"><i class="lg-dot"></i>Avg air temperature</span>`;

  const yrs = r.year_range
    ? `averaged over ${r.year_range[0]}–${r.year_range[1]}` : 'long-term average';
  $('resource-note').textContent =
    `${r.source}, ${yrs}, at ${r.lat}, ${r.lon}. `
    + `Tilting panels to ${summary.optimal_tilt_deg}° raises the annual figure to `
    + `${fmtInt(summary.annual_irradiation_kwh_m2)} kWh/m². Your system converts that `
    + `to ${fmtInt(summary.specific_yield_kwh_per_kwp)} kWh per kWp installed.`;
}

/** Grouped bars (GHI) + line (in-plane) + temperature dots, one SVG. */
function drawResourceChart(r, summary) {
  const M = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];
  const W = 380, H = 168, padL = 34, padR = 30, padT = 12, padB = 22;
  const iw = W - padL - padR, ih = H - padT - padB;
  const bw = iw / 12;

  const ghi = r.monthly_ghi_kwh_m2.map((v) => v ?? 0);
  const plane = summary.monthly_irradiation_kwh_m2 || [];
  const temps = r.monthly_temp_c || [];

  const maxIrr = Math.max(...ghi, ...plane.filter((v) => v != null), 1);
  const step = niceStep(maxIrr);
  const top = Math.ceil(maxIrr / step) * step;

  let grid = '';
  for (let v = 0; v <= top + 1e-9; v += step) {
    const y = padT + ih - (v / top) * ih;
    grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}"
                   stroke="#2a323d" stroke-width="1"/>
             <text x="${padL - 5}" y="${(y + 3.5).toFixed(1)}" text-anchor="end"
                   font-size="8.5" fill="#6e7c8c">${v}</text>`;
  }

  const bars = ghi.map((v, i) => {
    const h = (v / top) * ih;
    const x = padL + i * bw + bw * 0.2;
    return `<rect x="${x.toFixed(1)}" y="${(padT + ih - h).toFixed(1)}"
                  width="${(bw * 0.6).toFixed(1)}" height="${Math.max(h, 1).toFixed(1)}"
                  rx="2" fill="url(#gsun)">
              <title>${M[i]}: ${Math.round(v)} kWh/m² on the flat</title>
            </rect>
            <text x="${(padL + i * bw + bw / 2).toFixed(1)}" y="${H - 7}"
                  text-anchor="middle" font-size="8.5" fill="#6e7c8c">${M[i]}</text>`;
  }).join('');

  let planeLine = '';
  if (plane.length === 12) {
    const pts = plane.map((v, i) =>
      `${(padL + i * bw + bw / 2).toFixed(1)},${(padT + ih - (v / top) * ih).toFixed(1)}`);
    planeLine = `<polyline points="${pts.join(' ')}" fill="none" stroke="#5ac8fa"
                            stroke-width="1.8" stroke-linejoin="round"/>`
      + plane.map((v, i) => `<circle cx="${(padL + i * bw + bw / 2).toFixed(1)}"
            cy="${(padT + ih - (v / top) * ih).toFixed(1)}" r="2" fill="#5ac8fa">
            <title>${M[i]}: ${Math.round(v)} kWh/m² on tilted panels</title></circle>`).join('');
  }

  let tempMarks = '';
  const tv = temps.filter((v) => v != null);
  if (tv.length === 12) {
    const tMin = Math.min(...tv), tMax = Math.max(...tv);
    const span = Math.max(tMax - tMin, 1);
    tempMarks = temps.map((v, i) => {
      const y = padT + ih - ((v - tMin) / span) * (ih * 0.55) - ih * 0.05;
      return `<circle cx="${(padL + i * bw + bw / 2).toFixed(1)}" cy="${y.toFixed(1)}"
                      r="1.7" fill="none" stroke="#f85149" stroke-width="1.2">
                <title>${M[i]}: ${v}°C average</title></circle>`;
    }).join('');
    tempMarks += `<text x="${W - padR + 4}" y="${(padT + 8).toFixed(1)}"
                        font-size="8" fill="#f85149">${Math.round(tMax)}°</text>
                  <text x="${W - padR + 4}" y="${(padT + ih * 0.62).toFixed(1)}"
                        font-size="8" fill="#f85149">${Math.round(tMin)}°</text>`;
  }

  $('resource-chart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img"
         aria-label="Monthly solar exposure in kilowatt hours per square metre">
      <defs><linearGradient id="gsun" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ffcb5e"/><stop offset="100%" stop-color="#c2410c"/>
      </linearGradient></defs>
      ${grid}${bars}${planeLine}${tempMarks}
      <text x="2" y="9" font-size="8" fill="#6e7c8c">kWh/m²</text>
    </svg>`;
}

/* ───────────────────────── chart ───────────────────────── */

function drawChart(monthly) {
  const M = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];
  const W = 380, H = 150, padL = 40, padR = 8, padT = 12, padB = 22;
  const max = Math.max(...monthly, 1);
  const iw = W - padL - padR, ih = H - padT - padB;
  const bw = iw / 12;

  // y gridlines at 0, 50%, 100% of a rounded max
  const step = niceStep(max);
  const top = Math.ceil(max / step) * step;
  let grid = '';
  for (let v = 0; v <= top + 1e-9; v += step) {
    const y = padT + ih - (v / top) * ih;
    grid += `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}"
                   stroke="#2a323d" stroke-width="1"/>
             <text x="${padL - 6}" y="${(y + 3.5).toFixed(1)}" text-anchor="end"
                   font-size="9" fill="#6e7c8c">${v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v}</text>`;
  }

  const bars = monthly.map((v, i) => {
    const h = (v / top) * ih;
    const x = padL + i * bw + bw * 0.16;
    const y = padT + ih - h;
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(bw * 0.68).toFixed(1)}"
                  height="${Math.max(h, 1).toFixed(1)}" rx="2" fill="url(#g)">
              <title>${M[i]}: ${Math.round(v).toLocaleString()} kWh</title>
            </rect>
            <text x="${(padL + i * bw + bw / 2).toFixed(1)}" y="${H - 7}" text-anchor="middle"
                  font-size="9" fill="#6e7c8c">${M[i]}</text>`;
  }).join('');

  $('chart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Monthly generation in kilowatt hours">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#ffcb5e"/><stop offset="100%" stop-color="#e0761a"/>
        </linearGradient>
      </defs>
      ${grid}${bars}
      <text x="2" y="9" font-size="8.5" fill="#6e7c8c">kWh</text>
    </svg>`;
}

function niceStep(max) {
  const raw = max / 3;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const n = raw / mag;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
}

/* ───────────────────────── export ───────────────────────── */

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function stamp() {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
}

function exportGeoJSON() {
  if (!state.result) return;
  const fc = JSON.parse(JSON.stringify(state.result.geojson));
  fc.properties = {
    generated: new Date().toISOString(),
    summary: state.result.summary,
    assumptions: state.result.assumptions,
    imagery: state.result.imagery,
    model: state.result.model,
    disclaimer: 'Estimated from overhead imagery. Not a site survey.',
  };
  download(`rooftops-${stamp()}.geojson`, JSON.stringify(fc, null, 2),
           'application/geo+json');
}

function exportCSV() {
  if (!state.result) return;
  const s = state.result.summary, a = state.result.assumptions;
  const q = (v) => `"${String(v).replace(/"/g, '""')}"`;

  const lines = [
    '# Rooftop solar potential — estimate, not a site survey',
    `# generated,${new Date().toISOString()}`,
    `# model,${state.result.model.architecture}/${state.result.model.encoder}`,
    `# imagery,${state.result.imagery.provider} z${state.result.imagery.zoom}`,
    '',
    'metric,value,unit',
    `buildings,${s.building_count},count`,
    `roof_area,${s.roof_area_m2},m2`,
    `packing_factor,${a.packing_factor},ratio`,
    `usable_area,${s.usable_area_m2},m2`,
    `capacity,${s.capacity_kwp},kWp`,
    `annual_generation,${s.annual_kwh},kWh`,
    `annual_savings,${s.annual_savings},${a.currency}`,
    `net_cost,${s.net_cost},${a.currency}`,
    `payback,${s.payback_years ?? ''},years`,
    `co2_avoided,${s.co2_avoided_kg_per_year},kg/year`,
    '',
    'month,kwh',
    ...['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
      .map((m, i) => `${m},${s.monthly_kwh[i]}`),
    '',
    'roof_id,roof_area_m2,usable_area_m2,confidence,centroid_lon,centroid_lat',
    ...state.result.geojson.features.map((f) => {
      const r = f.geometry.coordinates[0];
      const cx = r.reduce((t, p) => t + p[0], 0) / r.length;
      const cy = r.reduce((t, p) => t + p[1], 0) / r.length;
      const p = f.properties;
      return [p.id, p.roof_area_m2, p.usable_area_m2, p.confidence,
              cx.toFixed(6), cy.toFixed(6)].map(q).join(',');
    }),
  ];
  download(`rooftops-${stamp()}.csv`, lines.join('\n'), 'text/csv');
}

/* ───────────────────────── about ───────────────────────── */

async function showAbout() {
  $('about-modal').hidden = false;
  const body = $('about-body');
  body.innerHTML = '<p>loading…</p>';
  try {
    const [m, a] = await Promise.all([
      (await fetch('/api/model')).json(),
      (await fetch('/api/assumptions')).json(),
    ]);
    const iou = m.metrics?.val?.iou;
    body.innerHTML = `
      <h3>What this does</h3>
      <p>A convolutional segmentation network looks at satellite imagery and marks
      every pixel it believes is a building roof. Those pixels are turned into
      polygons, measured on the WGS84 ellipsoid (not in Web Mercator, which would
      overstate area by <code>1/cos²(latitude)</code>), and run through a solar
      model to get energy, money and carbon.</p>

      <h3>The model</h3>
      <table>
        <tbody>
          <tr><td>Architecture</td><td><code>${esc(m.architecture)}</code> with a
              <code>${esc(m.encoder)}</code> encoder</td></tr>
          <tr><td>Trained on</td><td>${esc(m.trained_on || '—')}</td></tr>
          <tr><td>Held-out IoU</td><td class="metric-strong">${iou ?? '—'}</td></tr>
          <tr><td>Precision / recall</td><td>${m.metrics?.val?.precision ?? '—'} /
              ${m.metrics?.val?.recall ?? '—'}</td></tr>
          <tr><td>Input</td><td>${m.window}×${m.window} px at
              ${m.gsd_m_per_px} m/px (served at zoom ${m.serving_zoom})</td></tr>
          <tr><td>Runtime</td><td><code>${esc(m.runtime)}</code></td></tr>
        </tbody>
      </table>
      <p>IoU (intersection over union) is the overlap between predicted and true
      roof pixels. It is reported on cities the model never trained on. Accuracy
      is deliberately not the headline — only ~16% of pixels are roof, so a model
      that predicted "no roof" everywhere would already score ~84%.</p>

      <h3>Known limitations — please read</h3>
      <ul>${(m.limitations || []).map((l) => `<li>${esc(l)}</li>`).join('')}
        <li>The model outputs roof <b>outline</b>, not installable area. The
            packing factor bridges that gap and is a planning assumption, not a
            measurement.</li>
        <li>Nothing here models shading from trees or neighbouring buildings,
            roof pitch, structural capacity, or grid connection limits.</li>
      </ul>

      <h3>The calculation</h3>
      <ul>${a.chain.map((c) => `<li>${esc(c)}</li>`).join('')}</ul>

      <h3>Where the numbers come from</h3>
      <ul>${Object.entries(a.sources)
        .map(([k, v]) => `<li><code>${esc(k)}</code> — ${esc(v)}</li>`).join('')}</ul>

      <h3>Imagery</h3>
      <p>${state.cfg.tile_provider.attribution}. Served at zoom
      ${state.cfg.serving_zoom} so the ground resolution matches what the model
      was trained on. Check the provider's terms before deploying this publicly —
      some prohibit running ML over their tiles.</p>`;
  } catch (e) {
    body.innerHTML = `<p>Could not load model details: ${esc(e.message)}</p>`;
  }
}

/* ───────────────────────── utils ───────────────────────── */

function fmtInt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Math.round(v).toLocaleString();
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

boot();
