/* Industrial 3D SemSeg dataset site.
 * All data and media are fetched at runtime from the public HF dataset repo. */

const HF = "https://huggingface.co/datasets/min99ian/industrial-semseg-web/resolve/main";

const state = { meta: null, compare: null };

const $ = (sel) => document.querySelector(sel);

async function fetchJSON(path) {
  const r = await fetch(`${HF}/${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

function modelName(key) {
  return state.meta.models[key] || key;
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

  const noise = [
    ["noise/depth_clean.webp", "<b>Clean</b> depth"],
    ["noise/depth_mild.webp", "<b>Mild</b> corruption"],
    ["noise/depth_medium.webp", "<b>Medium</b> corruption"],
    ["noise/depth_strong.webp", "<b>Strong</b> corruption"],
    ["noise/normal_derived_clean.webp", "Normals — clean (derived)"],
    ["noise/normal_estimated_mild.webp", "Normals — mild (estimated)"],
    ["noise/normal_estimated_medium.webp", "Normals — medium (estimated)"],
    ["noise/normal_estimated_strong.webp", "Normals — strong (estimated)"],
  ];
  $("#noise-gallery").innerHTML = noise
    .map(([p, cap]) => `<figure class="fig"><img src="${HF}/media/${p}" loading="lazy" alt=""><figcaption>${cap}</figcaption></figure>`)
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

/* ---------- lightbox ---------- */

function setupLightbox() {
  const box = document.createElement("div");
  box.className = "lightbox";
  box.innerHTML = "<img alt=''>";
  document.body.appendChild(box);
  box.addEventListener("click", () => box.classList.remove("show"));
  document.addEventListener("click", (ev) => {
    const img = ev.target.closest(".fig img");
    if (!img) return;
    box.querySelector("img").src = img.src;
    box.classList.add("show");
  });
}

/* ---------- boot ---------- */

async function main() {
  [state.meta, state.compare] = await Promise.all([
    fetchJSON("metrics/meta.json"),
    fetchJSON("metrics/compare_manifest.json"),
  ]);
  renderDatasetGalleries();
  setupExplore();
  setupLightbox();
}

main().catch((err) => {
  console.error(err);
  document.querySelector("main").insertAdjacentHTML(
    "afterbegin",
    `<p class="loading">Failed to load data from Hugging Face: ${err.message}</p>`
  );
});
