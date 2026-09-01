/* Industrial 3D SemSeg benchmark site.
 * All data and media are fetched at runtime from the public HF dataset repo. */

const HF = "https://huggingface.co/datasets/min99ian/industrial-semseg-web/resolve/main";

const COLORS = {
  volt: "#2454ff",
  ditr: "#0e9f6e",
  litept: "#9333ea",
  ptv3: "#f59e0b",
  ptv3cac: "#d97706",
  oacnn: "#ef4444",
  octformer: "#0891b2",
  sonataft: "#64748b",
  sonatalin: "#a8a29e",
};

const state = { meta: null, lb: null, noise: null, cross: null, eff: null, compare: null };

const $ = (sel) => document.querySelector(sel);
const fmt = (x, d = 3) => (x == null ? "–" : x.toFixed(d));
const pct = (x) => (100 * x).toFixed(1);

async function fetchJSON(path) {
  const r = await fetch(`${HF}/${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

function modelName(key) {
  return state.meta.models[key] || key;
}

/* ---------- leaderboard ---------- */

function renderLeaderboard() {
  const tbody = $("#lb-table tbody");
  tbody.innerHTML = "";
  const core = new Set(state.meta.classes_17_core);
  state.lb.entries.forEach((e, i) => {
    const tr = document.createElement("tr");
    tr.className = `main-row rank-${i + 1}`;
    tr.innerHTML = `
      <td class="rank">${i + 1}</td>
      <td class="model">${modelName(e.model)}</td>
      <td class="num"><b>${fmt(e.miou17, 4)}</b></td>
      <td class="num">${fmt(e.miou23, 4)}</td>
      <td class="num">${fmt(e.macc, 4)}</td>
      <td class="num">${fmt(e.allacc, 4)}</td>
      <td><span class="chev">▶</span></td>`;
    const detail = document.createElement("tr");
    detail.className = "detail-row";
    detail.style.display = "none";
    const items = Object.entries(e.per_class_iou)
      .sort((a, b) => b[1] - a[1])
      .map(([c, v]) =>
        `<div class="pc-item ${core.has(c) ? "" : "excluded"}"><span>${c}</span><b>${fmt(v, 4)}</b></div>`)
      .join("");
    detail.innerHTML = `<td colspan="7"><div class="detail-inner">
        <div class="pc-grid">${items}</div>
        <p class="pc-note">Grayed-out classes are outside the 17-class evaluation scope
        (rare or ambiguous under single-view captures).</p>
      </div></td>`;
    tr.addEventListener("click", () => {
      const open = detail.style.display !== "none";
      detail.style.display = open ? "none" : "";
      tr.classList.toggle("open", !open);
    });
    tbody.append(tr, detail);
  });
}

/* ---------- robustness charts ---------- */

const AXIS = { color: "#8a94a6", font: { family: "Inter", size: 12 } };
const GRID = { color: "#eef1f6" };

function lineDataset(key, values) {
  return {
    label: modelName(key),
    data: values,
    borderColor: COLORS[key],
    backgroundColor: COLORS[key],
    borderWidth: 2.2,
    pointRadius: 3.5,
    tension: 0.25,
  };
}

function renderNoiseCharts() {
  const levels = state.noise.levels.map((l) => l[0].toUpperCase() + l.slice(1));
  const order = state.lb.entries.map((e) => e.model);
  new Chart($("#noise-chart"), {
    type: "line",
    data: {
      labels: levels,
      datasets: order.map((k) => lineDataset(k, state.noise.series[k])),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12, font: AXIS.font, color: "#4b5565" } },
        title: { display: true, text: "Frame-averaged 17-class mIoU vs. depth corruption", color: "#101623", font: { family: "Inter", size: 14, weight: "600" } },
        tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt(c.parsed.y, 3)}` } },
      },
      scales: {
        y: { min: 0, max: 1, ticks: AXIS, grid: GRID },
        x: { ticks: AXIS, grid: { display: false } },
      },
    },
  });

  const mix = state.noise.ptv3_mixed_noise_training;
  new Chart($("#mix-chart"), {
    type: "line",
    data: {
      labels: levels,
      datasets: [
        { ...lineDataset("ptv3", mix.ptv3_clean_trained), label: "PTv3 — clean-trained" },
        { ...lineDataset("volt", mix.ptv3_mix_403030_trained), label: "PTv3 — mixed-noise-trained (40/30/30)", borderDash: [6, 4] },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12, font: AXIS.font, color: "#4b5565" } },
        title: { display: true, text: "Mixed-noise training (PTv3)", color: "#101623", font: { family: "Inter", size: 14, weight: "600" } },
      },
      scales: {
        y: { min: 0, max: 1, ticks: AXIS, grid: GRID },
        x: { ticks: AXIS, grid: { display: false } },
      },
    },
  });
}

function renderNoiseGallery() {
  const figs = [
    ["noise/depth_clean.webp", "<b>Clean</b> depth"],
    ["noise/depth_mild.webp", "<b>Mild</b> corruption"],
    ["noise/depth_medium.webp", "<b>Medium</b> corruption"],
    ["noise/depth_strong.webp", "<b>Strong</b> corruption"],
    ["noise/normal_derived_clean.webp", "Normals — clean (derived)"],
    ["noise/normal_estimated_mild.webp", "Normals — mild (estimated)"],
    ["noise/normal_estimated_medium.webp", "Normals — medium (estimated)"],
    ["noise/normal_estimated_strong.webp", "Normals — strong (estimated)"],
  ];
  $("#noise-gallery").innerHTML = figs
    .map(([p, cap]) => `<figure class="fig"><img src="${HF}/media/${p}" loading="lazy" alt=""><figcaption>${cap}</figcaption></figure>`)
    .join("");
}

/* ---------- cross-robot ---------- */

function heatColor(v) {
  // 0 -> red-ish, 1 -> green-ish, light backgrounds
  const h = 8 + 130 * Math.max(0, Math.min(1, v));
  return `hsl(${h} 82% 94%)`;
}

function renderCrossRobot(metric = "miou17") {
  const robots = state.cross.robots;
  const order = state.lb.entries.map((e) => e.model);
  $("#cr-table thead").innerHTML =
    `<tr><th>Model</th>${robots.map((r) => `<th class="num">${r}</th>`).join("")}</tr>`;
  $("#cr-table tbody").innerHTML = order
    .map((k) => {
      const cells = robots
        .map((r) => {
          const v = state.cross.series[k]?.[r]?.[metric];
          return `<td class="heat" style="background:${heatColor(v)}">${fmt(v, 3)}</td>`;
        })
        .join("");
      return `<tr><td class="model">${modelName(k)}</td>${cells}</tr>`;
    })
    .join("");
}

/* ---------- interactive prediction explorer ---------- */

const explore = { frame: null, a: "volt", b: "ditr", mode: "pred" };

function frameURL(frameId, file) {
  return `${HF}/media/compare/${frameId}/${file}`;
}

function renderFrameList() {
  const groups = {};
  state.compare.frames.forEach((f) => (groups[f.group] ??= []).push(f));
  $("#frame-list").innerHTML = Object.entries(groups)
    .map(([g, frames]) => `
      <div class="frame-group">${g}</div>
      ${frames.map((f) => `
        <button class="frame-item ${f.id === explore.frame ? "active" : ""}" data-frame="${f.id}" type="button">
          <img src="${frameURL(f.id, "thumb.webp")}" loading="lazy" alt="">
          <span>${f.label}</span>
        </button>`).join("")}`)
    .join("");
}

function renderModelSelects() {
  const opts = (sel) => Object.entries(state.meta.models)
    .map(([k, name]) => `<option value="${k}" ${k === explore[sel] ? "selected" : ""}>${name}</option>`)
    .join("");
  $("#model-a").innerHTML = opts("a");
  $("#model-b").innerHTML = opts("b");
}

function renderPanels() {
  const f = explore.frame;
  const suffix = explore.mode;
  const panels = [
    ["rgb.webp", "RGB input"],
    ["gt.webp", "Ground truth"],
    [`${explore.a}_${suffix}.webp`, `${modelName(explore.a)} — ${suffix === "pred" ? "prediction" : "error map"}`],
    [`${explore.b}_${suffix}.webp`, `${modelName(explore.b)} — ${suffix === "pred" ? "prediction" : "error map"}`],
  ];
  $("#panel-grid").innerHTML = panels
    .map(([file, cap]) =>
      `<figure class="fig"><img src="${frameURL(f, file)}" alt="${cap}"><figcaption><b>${cap}</b></figcaption></figure>`)
    .join("");
  renderExploreLegend();
}

function chip(rgb, label) {
  return `<span class="chip"><i style="background:rgb(${rgb.join(",")})"></i>${label}</span>`;
}

function renderExploreLegend() {
  const m = state.compare;
  if (explore.mode === "err") {
    $("#explore-legend").innerHTML =
      chip(m.error_palette.correct, "correct") +
      chip(m.error_palette.wrong, "misclassified") +
      chip(m.error_palette.ignored, "ignored");
    return;
  }
  const f = m.frames.find((x) => x.id === explore.frame);
  $("#explore-legend").innerHTML = f.classes_present
    .map((c) => chip(m.class_palette[c], c.replaceAll("_", " ")))
    .join("");
}

function setFrame(id) {
  explore.frame = id;
  document.querySelectorAll(".frame-item").forEach((el) =>
    el.classList.toggle("active", el.dataset.frame === id));
  renderPanels();
}

function feelingLucky() {
  const frames = state.compare.frames;
  const models = Object.keys(state.meta.models);
  explore.frame = frames[Math.floor(Math.random() * frames.length)].id;
  const a = models[Math.floor(Math.random() * models.length)];
  let b = a;
  while (b === a) b = models[Math.floor(Math.random() * models.length)];
  explore.a = a;
  explore.b = b;
  renderModelSelects();
  setFrame(explore.frame);
}

function setupExplore() {
  explore.frame = state.compare.frames[0].id;
  renderFrameList();
  renderModelSelects();
  renderPanels();

  $("#frame-list").addEventListener("click", (ev) => {
    const item = ev.target.closest(".frame-item");
    if (item) setFrame(item.dataset.frame);
  });
  $("#model-a").addEventListener("change", (ev) => { explore.a = ev.target.value; renderPanels(); });
  $("#model-b").addEventListener("change", (ev) => { explore.b = ev.target.value; renderPanels(); });
  $("#ex-mode").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    document.querySelectorAll("#ex-mode button").forEach((b) => b.classList.toggle("active", b === btn));
    explore.mode = btn.dataset.mode;
    renderPanels();
  });
  $("#lucky").addEventListener("click", feelingLucky);
}

/* ---------- efficiency ---------- */

function renderEfficiency() {
  const entries = state.eff.entries;
  const miou = Object.fromEntries(state.lb.entries.map((e) => [e.model, e.miou17]));
  const keys = Object.keys(entries);

  new Chart($("#eff-chart"), {
    type: "bubble",
    data: {
      datasets: keys.map((k) => ({
        label: modelName(k),
        data: [{ x: entries[k].latency_ms_mean, y: miou[k], r: 6 + 9 * Math.sqrt(entries[k].peak_vram_gib) }],
        backgroundColor: COLORS[k] + "cc",
        borderColor: COLORS[k],
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 12, font: AXIS.font, color: "#4b5565" } },
        tooltip: {
          callbacks: {
            label: (c) =>
              ` ${c.dataset.label}: ${fmt(c.parsed.y, 3)} mIoU · ${c.parsed.x} ms · ${entries[keys[c.datasetIndex]].peak_vram_gib} GiB`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "Forward latency (ms / frame)", color: "#4b5565", font: AXIS.font }, ticks: AXIS, grid: GRID },
        y: { title: { display: true, text: "Clean 17-class mIoU", color: "#4b5565", font: AXIS.font }, ticks: AXIS, grid: GRID },
      },
    },
  });

  $("#eff-table tbody").innerHTML = keys
    .sort((a, b) => entries[a].latency_ms_mean - entries[b].latency_ms_mean)
    .map((k) => `<tr>
        <td class="model">${modelName(k)}</td>
        <td class="num">${entries[k].latency_ms_mean.toFixed(1)} ± ${entries[k].latency_ms_std.toFixed(1)}</td>
        <td class="num">${entries[k].latency_ms_p90.toFixed(1)}</td>
        <td class="num">${entries[k].peak_vram_gib.toFixed(2)}</td>
      </tr>`)
    .join("");
}

/* ---------- dataset galleries ---------- */

function renderDatasetGalleries() {
  const mods = [
    ["modalities/rgb_exp.webp", "RGB"],
    ["modalities/depth_exp.webp", "Depth"],
    ["modalities/normal_exp.webp", "Surface normals"],
    ["modalities/sem_seg_gt_exp.webp", "Semantic ground truth"],
  ];
  $("#modality-gallery").innerHTML = mods
    .map(([p, cap]) => `<figure class="fig"><img src="${HF}/media/${p}" loading="lazy" alt="${cap}"><figcaption><b>${cap}</b></figcaption></figure>`)
    .join("");

  const samp = [
    ["sampling/scene-level-sampling-cropped.webp", "Scene-level sampling"],
    ["sampling/object-centric-sampling-1.5m-cropped.webp", "Object-centric — 1.5 m"],
    ["sampling/object-centric-sampling-2.0m-cropped.webp", "Object-centric — 2.0 m"],
    ["sampling/object-centric-sampling-2.5m-cropped.webp", "Object-centric — 2.5 m"],
  ];
  $("#sampling-gallery").innerHTML = samp
    .map(([p, cap]) => `<figure class="fig"><img src="${HF}/media/${p}" loading="lazy" alt="${cap}"><figcaption><b>${cap}</b></figcaption></figure>`)
    .join("");
}

/* ---------- lightbox ---------- */

function setupLightbox() {
  const box = document.createElement("div");
  box.className = "lightbox";
  box.innerHTML = "<img alt=''>";
  document.body.appendChild(box);
  box.addEventListener("click", () => box.classList.remove("show"));
  document.addEventListener("click", (ev) => {
    const img = ev.target.closest(".fig img, .legend-row img");
    if (!img) return;
    box.querySelector("img").src = img.src;
    box.classList.add("show");
  });
}

/* ---------- boot ---------- */

async function main() {
  [state.meta, state.lb, state.noise, state.cross, state.eff, state.compare] = await Promise.all([
    fetchJSON("metrics/meta.json"),
    fetchJSON("metrics/leaderboard_clean.json"),
    fetchJSON("metrics/noise_robustness.json"),
    fetchJSON("metrics/cross_robot.json"),
    fetchJSON("metrics/efficiency.json"),
    fetchJSON("metrics/compare_manifest.json"),
  ]);
  renderLeaderboard();
  renderNoiseCharts();
  renderNoiseGallery();
  renderCrossRobot();
  setupExplore();
  renderEfficiency();
  renderDatasetGalleries();
  setupLightbox();

  $("#cr-metric").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    document.querySelectorAll("#cr-metric button").forEach((b) => b.classList.toggle("active", b === btn));
    renderCrossRobot(btn.dataset.metric);
  });
}

main().catch((err) => {
  console.error(err);
  document.querySelector("main").insertAdjacentHTML(
    "afterbegin",
    `<p class="loading">Failed to load benchmark data from Hugging Face: ${err.message}</p>`
  );
});
