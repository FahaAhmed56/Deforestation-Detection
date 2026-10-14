/* ForestWatch front-end — plain JavaScript + Leaflet (no build step). */
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmtHa = (v) => (v >= 1000 ? (v / 1000).toFixed(2) + "k" : (+v).toFixed(v < 10 ? 2 : 1)) + " ha";
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

/* ------------------------------------------------------------------ map */
const map = L.map("map", { zoomControl: true, attributionControl: true }).setView([-9.95, -63.1], 12);
const basemaps = {
  "Satellite (Esri)": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 18, attribution: "Imagery © Esri" }),
  "Streets (OSM)": L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 19, attribution: "© OpenStreetMap contributors" }),
  "Dark": L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    { maxZoom: 19, attribution: "© CARTO © OSM" }),
};
basemaps["Satellite (Esri)"].addTo(map);
L.control.layers(basemaps, null, { position: "topright" }).addTo(map);
L.control.scale({ imperial: false }).addTo(map);

const state = { run: null, interval: 0, layers: {}, geo: null, bbox: null, swipeX: null };
const drawn = new L.FeatureGroup().addTo(map);

/* ---------------------------------------------------------------- tabs */
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + t.dataset.tab));
  if (t.dataset.tab === "model") loadModel();
}));

/* -------------------------------------------------------------- health */
async function health() {
  const el = $("#modelStatus");
  try {
    const h = await api("/api/health");
    el.className = "status " + (h.model_trained ? "ok" : "bad");
    el.lastChild.textContent = h.model_trained ? ` model ready · v${h.version}` : " model not trained — run python -m src.train";
  } catch { el.className = "status bad"; el.lastChild.textContent = " server offline"; }
}

/* ---------------------------------------------------------------- runs */
async function loadRuns(selectId) {
  const runs = await api("/api/runs");
  const sel = $("#runSelect");
  sel.innerHTML = runs.map((r) => `<option value="${r.id}">${r.name} · ${r.dates[0]} → ${r.dates.at(-1)}</option>`).join("");
  $("#runEmpty").classList.toggle("hidden", runs.length > 0);
  $("#runBody").classList.toggle("hidden", runs.length === 0);
  if (runs.length) {
    sel.value = selectId && runs.some((r) => r.id === selectId) ? selectId : runs[0].id;
    await openRun(sel.value);
  }
}
$("#runSelect").addEventListener("change", (e) => openRun(e.target.value));

async function openRun(id) {
  const run = await api(`/api/runs/${id}`);
  state.run = run;
  state.interval = Math.max(0, run.changes.length - 1);
  const t = run.totals;
  $("#runMeta").innerHTML = `<span class="tag">${run.source}</span><span class="tag">${run.scenes.length} dates</span>
    <span class="tag">${run.shape[1]}×${run.shape[0]} px</span><br>
    ${run.scenes.map((s) => `${s.date} <span title="scene">· ${s.scene_id}</span> · ${s.aoi_invalid_pct ?? 0}% cloud`).join("<br>")}`;
  $("#sForestStart").textContent = fmtHa(t.forest_start_ha);
  $("#sForestEnd").textContent = fmtHa(t.forest_end_ha);
  $("#sLoss").textContent = fmtHa(t.loss_ha);
  $("#sGain").textContent = fmtHa(t.gain_ha);
  $("#intervalSelect").innerHTML = run.changes
    .map((c, i) => `<option value="${i}">${c.from} → ${c.to}  (−${c.loss_ha} ha)</option>`).join("");
  $("#intervalSelect").value = state.interval;
  drawChart(run);
  map.fitBounds(run.bounds, { padding: [20, 20] });
  await showInterval();
}
$("#intervalSelect").addEventListener("change", async (e) => { state.interval = +e.target.value; await showInterval(); });

function clearLayers() {
  Object.values(state.layers).forEach((l) => l && map.removeLayer(l));
  state.layers = {};
}

async function showInterval() {
  const run = state.run, c = run.changes[state.interval];
  if (!c) return;
  clearLayers();
  const base = `/runs/${run.id}/`, b = run.bounds;
  const before = run.scenes.find((s) => s.date === c.from), after = run.scenes.find((s) => s.date === c.to);
  const op = +$("#opacity").value / 100;
  state.layers.before = L.imageOverlay(base + before.rgb, b, { opacity: op });
  state.layers.after = L.imageOverlay(base + after.rgb, b, { opacity: op });
  state.layers.forest = L.imageOverlay(base + after.forest, b, { opacity: op });
  state.layers.gain = L.imageOverlay(base + c.gain, b, { opacity: op });
  state.layers.loss = L.imageOverlay(base + c.loss, b, { opacity: op });

  state.geo = await api(base + c.geojson).catch(() => ({ features: [] }));
  state.layers.poly = L.geoJSON(state.geo, {
    filter: (f) => f.properties.type === "loss",
    style: { color: "#ffdd57", weight: 1.4, fill: false, dashArray: "3 2" },
    onEachFeature: (f, l) => l.bindPopup(popup(f.properties)),
  });
  state.layers.bounds = L.rectangle(b, { color: "#3fb950", weight: 1, fill: false, dashArray: "6 4", interactive: false });
  state.layers.bounds.addTo(map);

  const s = c.loss_pct_of_forest;
  $("#intervalStats").innerHTML = `
    <div><b style="color:var(--loss)">${fmtHa(c.loss_ha)}</b><span>forest lost</span></div>
    <div><b>${c.loss_patches}</b><span>clearings</span></div>
    <div><b>${s}%</b><span>of forest</span></div>`;
  $("#dateBadge").innerHTML = `<b>${c.from}</b> → <b>${c.to}</b>`;
  $("#dateBadge").classList.remove("hidden");
  $("#dlGeojson").href = base + c.geojson;
  $("#dlGeojson").download = `${run.id}_${c.from}_${c.to}.geojson`;
  $("#dlSummary").href = base + c.summary;
  renderPatches();
  renderTruth(c);
  applyLayers();
}

function popup(p) {
  return `<div><b>${p.type === "loss" ? "Cleared area" : "Regrowth"} #${p.id}</b><br>
    Area: <strong>${p.area_ha} ha</strong> (${p.pixels} px)<br>
    Detected between ${p.from} and ${p.to}<br>
    Confidence (prob. drop): ${(p.confidence * 100).toFixed(0)}%<br>
    Centroid: ${p.centroid[0].toFixed(5)}, ${p.centroid[1].toFixed(5)}</div>`;
}

function renderPatches() {
  const loss = state.geo.features.filter((f) => f.properties.type === "loss");
  $("#patchCount").textContent = `(${loss.length} total)`;
  $("#patchRows").innerHTML = loss.slice(0, 12).map((f) => {
    const p = f.properties;
    return `<tr data-id="${p.id}"><td>${p.id}</td><td>${p.area_ha} ha</td><td>${(p.confidence * 100).toFixed(0)}%</td>
      <td>${p.centroid[0].toFixed(4)}, ${p.centroid[1].toFixed(4)}</td></tr>`;
  }).join("") || `<tr><td colspan="4" style="color:var(--muted)">No clearings above the minimum size.</td></tr>`;
  $$("#patchRows tr[data-id]").forEach((tr) => tr.addEventListener("click", () => {
    const id = +tr.dataset.id;
    state.layers.poly.eachLayer((l) => {
      if (l.feature.properties.id === id) {
        if (!map.hasLayer(state.layers.poly)) { $("#lyPoly").checked = true; applyLayers(); }
        map.fitBounds(l.getBounds(), { maxZoom: 16, padding: [60, 60] });
        l.openPopup();
      }
    });
  }));
}

function renderTruth(c) {
  const box = $("#truthBox"), a = c.accuracy_vs_truth;
  box.classList.toggle("hidden", !a);
  if (a) box.innerHTML = `<b>Scored against known truth</b> (synthetic demo)<br>
    Precision ${(a.precision * 100).toFixed(1)}% · Recall ${(a.recall * 100).toFixed(1)}% ·
    F1 ${(a.f1 * 100).toFixed(1)}% · IoU ${(a.iou * 100).toFixed(1)}%`;
}

/* --------------------------------------------------------------- layers */
const layerMap = { lyBefore: "before", lyAfter: "after", lyForest: "forest", lyGain: "gain", lyLoss: "loss", lyPoly: "poly" };
function applyLayers() {
  const swipe = $("#lySwipe").checked;
  for (const [cb, key] of Object.entries(layerMap)) {
    const on = $("#" + cb).checked || (swipe && (key === "before" || key === "after"));
    const l = state.layers[key];
    if (!l) continue;
    if (on && !map.hasLayer(l)) l.addTo(map);
    if (!on && map.hasLayer(l)) map.removeLayer(l);
  }
  // keep a sensible stacking order
  ["before", "after", "forest", "gain", "loss"].forEach((k) => state.layers[k] && map.hasLayer(state.layers[k]) && state.layers[k].bringToFront());
  state.layers.poly && map.hasLayer(state.layers.poly) && state.layers.poly.bringToFront();
  $("#swipe").classList.toggle("hidden", !swipe);
  updateSwipe();
}
Object.keys(layerMap).concat("lySwipe").forEach((id) => $("#" + id).addEventListener("change", applyLayers));
$("#opacity").addEventListener("input", (e) => {
  $("#opVal").textContent = e.target.value + "%";
  ["before", "after", "forest", "gain", "loss"].forEach((k) => state.layers[k]?.setOpacity(e.target.value / 100));
});

/* ---------------------------------------------------- swipe comparison */
function updateSwipe() {
  const on = !$("#lySwipe").checked ? false : true;
  const w = map.getSize().x;
  if (state.swipeX == null) state.swipeX = w / 2;
  $("#swipe").style.left = state.swipeX + "px";
  ["after", "loss", "gain", "forest"].forEach((k) => {
    const el = state.layers[k]?.getElement?.();
    if (!el) return;
    if (!on) { el.style.clipPath = ""; return; }
    const left = map.latLngToContainerPoint(state.layers[k].getBounds().getNorthWest()).x;
    el.style.clipPath = `inset(0 0 0 ${Math.max(0, state.swipeX - left)}px)`;
  });
}
map.on("move zoom zoomend resize", updateSwipe);
(() => {
  const s = $("#swipe"); let drag = false;
  s.addEventListener("pointerdown", (e) => { drag = true; s.setPointerCapture(e.pointerId); map.dragging.disable(); });
  s.addEventListener("pointerup", () => { drag = false; map.dragging.enable(); });
  s.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const r = $("#map").getBoundingClientRect();
    state.swipeX = Math.min(Math.max(e.clientX - r.left, 0), r.width);
    updateSwipe();
  });
})();

/* ---------------------------------------------------------------- chart */
function drawChart(run) {
  const W = 350, H = 180, P = { l: 44, r: 40, t: 26, b: 26 };
  const d = run.scenes.map((s) => s.date), f = run.scenes.map((s) => s.forest_ha);
  const loss = [0, ...run.changes.map((c) => c.loss_ha)];
  const fMin = Math.min(...f) * 0.97, fMax = Math.max(...f) * 1.01, lMax = Math.max(...loss, 1);
  const x = (i) => P.l + (d.length === 1 ? 0 : (i * (W - P.l - P.r)) / (d.length - 1));
  const yF = (v) => P.t + (1 - (v - fMin) / (fMax - fMin || 1)) * (H - P.t - P.b);
  const yL = (v) => H - P.b - (v / lMax) * (H - P.t - P.b) * 0.8;
  const bars = loss.map((v, i) => i === 0 ? "" :
    `<rect x="${x(i) - 11}" y="${yL(v)}" width="22" height="${H - P.b - yL(v)}" fill="var(--loss)" opacity=".75" rx="3"><title>${v} ha lost</title></rect>
     <text x="${x(i)}" y="${yL(v) - 4}" text-anchor="middle" style="fill:var(--loss)">−${Math.round(v)}</text>`).join("");
  const line = f.map((v, i) => `${i ? "L" : "M"}${x(i)},${yF(v)}`).join("");
  const dots = f.map((v, i) => `<circle cx="${x(i)}" cy="${yF(v)}" r="4" fill="var(--accent)"><title>${v} ha forest</title></circle>`).join("");
  const labels = d.map((v, i) => `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${v.slice(0, 7)}</text>`).join("");
  $("#chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}">
    <line x1="${P.l}" x2="${W - P.r}" y1="${H - P.b}" y2="${H - P.b}" stroke="var(--line)"/>
    ${bars}<path d="${line}" fill="none" stroke="var(--accent)" stroke-width="2.5"/>${dots}${labels}
    <text x="4" y="${yF(fMax)}" >${Math.round(fMax)}</text><text x="4" y="${yF(fMin) }">${Math.round(fMin)}</text>
    <text x="${W - 4}" y="10" text-anchor="end" style="fill:var(--loss)">■ loss in interval (ha)</text>
    <text x="4" y="10" style="fill:var(--accent)">● forest area (ha)</text></svg>`;
}

/* ----------------------------------------------------------- new analysis */
const today = new Date().toISOString().slice(0, 10);
function addDate(v = "") {
  const row = document.createElement("div");
  row.className = "daterow";
  row.innerHTML = `<input type="date" max="${today}" min="2017-04-01" value="${v}"><button title="remove">✕</button>`;
  row.querySelector("button").onclick = () => $$(".daterow").length > 2 && row.remove();
  $("#dateList").appendChild(row);
}
addDate("2023-07-15"); addDate("2025-07-15");
$("#addDate").onclick = () => addDate();

$$("input[name=src]").forEach((r) => r.addEventListener("change", () => {
  const demo = $("input[name=src]:checked").value === "demo";
  $("#sentinelForm").classList.toggle("hidden", demo);
  $("#demoForm").classList.toggle("hidden", !demo);
}));

const rectDrawer = new L.Draw.Rectangle(map, { shapeOptions: { color: "#3fb950", weight: 2, fillOpacity: 0.08 } });
$("#drawBtn").onclick = () => { $$(".tab")[1].click(); rectDrawer.enable(); };
map.on(L.Draw.Event.CREATED, (e) => setBbox(e.layer.getBounds()));

function setBbox(bounds) {
  drawn.clearLayers();
  L.rectangle(bounds, { color: "#3fb950", weight: 2, fillOpacity: 0.08 }).addTo(drawn);
  const b = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()].map((v) => +v.toFixed(5));
  const kmW = (b[2] - b[0]) * 111.32 * Math.cos(((b[1] + b[3]) / 2) * Math.PI / 180), kmH = (b[3] - b[1]) * 110.57;
  state.bbox = b;
  $("#bboxText").innerHTML = `[${b.join(", ")}]<br>${kmW.toFixed(1)} × ${kmH.toFixed(1)} km` +
    (Math.max(kmW, kmH) > 15 ? ` — <span style="color:var(--loss)">too large, keep under 15 km</span>` : "");
}

const PRESETS = {
  rondonia: { name: "Rondônia, Brazil", b: [-63.20, -10.00, -63.06, -9.88], d: ["2021-07-20", "2024-07-20"] },
  para: { name: "Pará BR-163", b: [-55.10, -6.60, -54.96, -6.48], d: ["2021-08-01", "2024-08-01"] },
  borneo: { name: "Central Kalimantan", b: [113.80, -2.30, 113.94, -2.17], d: ["2019-08-15", "2024-08-15"] },
};
$$(".chip").forEach((c) => c.addEventListener("click", () => {
  const p = PRESETS[c.dataset.preset];
  const bounds = L.latLngBounds([p.b[1], p.b[0]], [p.b[3], p.b[2]]);
  setBbox(bounds); map.fitBounds(bounds);
  $("#runName").value = p.name;
  $("#dateList").innerHTML = ""; p.d.forEach(addDate);
}));

$("#runBtn").onclick = async () => {
  const demo = $("input[name=src]:checked").value === "demo";
  const body = demo ? { source: "demo" } : {
    source: "sentinel", name: $("#runName").value || "My area", bbox: state.bbox,
    dates: $$(".daterow input").map((i) => i.value).filter(Boolean),
  };
  const con = $("#console"), btn = $("#runBtn");
  con.classList.remove("hidden"); con.textContent = "Submitting…\n";
  btn.disabled = true;
  try {
    let job = await api("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    while (job.status === "queued" || job.status === "running") {
      await new Promise((r) => setTimeout(r, 1200));
      job = await api(`/api/jobs/${job.id}`);
      con.textContent = job.log.join("\n") || job.status + "…";
      con.scrollTop = con.scrollHeight;
    }
    if (job.status === "error") throw new Error(job.error);
    con.textContent += "\n✓ Finished — opening results.";
    drawn.clearLayers();
    $$(".tab")[0].click();
    await loadRuns(job.run_id);
  } catch (err) {
    con.textContent += `\n✗ ${err.message}`;
  } finally { btn.disabled = false; }
};

$("#delRun").onclick = async () => {
  if (!state.run || !confirm(`Delete analysis "${state.run.name}"?`)) return;
  await api(`/api/runs/${state.run.id}`, { method: "DELETE" });
  clearLayers(); await loadRuns();
};

/* ---------------------------------------------------------------- model */
async function loadModel() {
  const m = await api("/api/model").catch(() => null);
  if (!m || !m.history) { $("#modelStats").innerHTML = `<div class="empty">Model not trained yet.</div>`; return; }
  const t = m.test_metrics || {}, h = m.history, last = h.at(-1);
  const pct = (v) => (v == null ? "–" : (v * 100).toFixed(1) + "%");
  $("#modelStats").innerHTML = `
    <div class="stat"><span>Test IoU</span><b>${pct(t.iou)}</b></div>
    <div class="stat"><span>Test F1</span><b>${pct(t.f1)}</b></div>
    <div class="stat"><span>Test accuracy</span><b>${pct(t.accuracy)}</b></div>
    <div class="stat"><span>Epochs · base width</span><b>${h.length} · ${m.config.base_channels}</b></div>`;
  const W = 350, H = 140, P = 30, ious = h.map((r) => r.val_iou), lo = Math.min(...ious) - 0.01, hi = 1;
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(h.length - 1, 1), y = (v) => 10 + (1 - (v - lo) / (hi - lo)) * (H - 35);
  $("#iouChart").innerHTML = `<svg viewBox="0 0 ${W} ${H}">
    <path d="${ious.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("")}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    ${ious.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="var(--accent)"><title>epoch ${i + 1}: ${v.toFixed(4)}</title></circle>`).join("")}
    <text x="${P}" y="${H - 6}">epoch 1</text><text x="${W - P}" y="${H - 6}" text-anchor="end">epoch ${h.length}</text>
    <text x="2" y="${y(last.val_iou) + 3}">${last.val_iou.toFixed(3)}</text></svg>`;
}

/* ----------------------------------------------------------------- boot */
health();
loadRuns().catch((e) => console.error(e));
