/* Interactive four-view point cloud comparison.
 * Geometry and labels are fetched as binaries from the public HF dataset repo;
 * the four views share one camera and differ only in per-point colors. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const HF = "https://huggingface.co/datasets/min99ian/industrial-semseg-web/resolve/main";

const MODELS = {
  ptv3: "PTv3",
  litept: "LitePT",
  oacnn: "OA-CNN",
  ptv3cac: "PTv3+CAC",
  octformer: "OctFormer",
  sonatalin: "Sonata (linear probing)",
  sonataft: "Sonata (fine-tuned)",
  ditr: "DiTR",
  volt: "VoLT",
};

const state = {
  manifest: null,
  frame: null,
  a: "volt",
  b: "ditr",
  mode: "pred",
  // per-frame data
  positions: null, rgb: null, gt: null,
  preds: new Map(), // model -> Int8Array (for current frame)
};

const $ = (s) => document.querySelector(s);

/* ---------- colors ---------- */

function hashColor(id) {
  if (id < 0) return [40, 40, 40];
  if (id === 0) return [20, 20, 20];
  return [(id * 37 + 17) % 255, (id * 67 + 29) % 255, (id * 97 + 43) % 255];
}

const CLASS_COLORS = new Float32Array(24 * 3);
for (let i = -1; i < 23; i++) {
  const [r, g, b] = hashColor(i);
  CLASS_COLORS.set([r / 255, g / 255, b / 255], (i + 1) * 3);
}
const ERR = {
  correct: [120 / 255, 120 / 255, 120 / 255],
  wrong: [229 / 255, 57 / 255, 53 / 255],
  ignored: [0, 0, 0],
};

function semanticColors(labels) {
  const n = labels.length;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const o = (labels[i] + 1) * 3;
    out[i * 3] = CLASS_COLORS[o];
    out[i * 3 + 1] = CLASS_COLORS[o + 1];
    out[i * 3 + 2] = CLASS_COLORS[o + 2];
  }
  return out;
}

function errorColors(gt, pred) {
  const n = gt.length;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = gt[i] < 0 ? ERR.ignored : gt[i] === pred[i] ? ERR.correct : ERR.wrong;
    out[i * 3] = c[0]; out[i * 3 + 1] = c[1]; out[i * 3 + 2] = c[2];
  }
  return out;
}

function rgbColors(rgb) {
  const n = rgb.length / 3;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) out[i] = rgb[i] / 255;
  return out;
}

/* ---------- data loading ---------- */

async function fetchBin(frame, file) {
  const r = await fetch(`${HF}/media/pc/${frame}/${file}`);
  if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

async function loadFrame(frameId) {
  const f = state.manifest.frames.find((x) => x.id === frameId);
  const [coordBuf, rgbBuf, gtBuf] = await Promise.all([
    fetchBin(frameId, "coord.bin"),
    fetchBin(frameId, "rgb.bin"),
    fetchBin(frameId, "gt.bin"),
  ]);
  const q = new Uint16Array(coordBuf);
  const n = q.length / 3;
  const pos = new Float32Array(n * 3);
  const [mnx, mny, mnz] = f.bbox_min;
  const sx = (f.bbox_max[0] - mnx) / 65535, sy = (f.bbox_max[1] - mny) / 65535, sz = (f.bbox_max[2] - mnz) / 65535;
  for (let i = 0; i < n; i++) {
    pos[i * 3] = mnx + q[i * 3] * sx;
    pos[i * 3 + 1] = mny + q[i * 3 + 1] * sy;
    pos[i * 3 + 2] = mnz + q[i * 3 + 2] * sz;
  }
  state.positions = pos;
  state.rgb = new Uint8Array(rgbBuf);
  state.gt = new Int8Array(gtBuf);
  state.preds.clear();
}

async function loadPred(model) {
  if (!state.preds.has(model)) {
    const buf = await fetchBin(state.frame, `${model}.bin`);
    state.preds.set(model, new Int8Array(buf));
  }
  return state.preds.get(model);
}

/* ---------- three.js four-view rig ---------- */

const rig = { renderer: null, camera: null, controls: null, views: [] };

function setupViews() {
  const container = $("#views");
  rig.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "high-performance" });
  rig.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  rig.renderer.domElement.className = "views-gl";
  container.prepend(rig.renderer.domElement);

  rig.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
  rig.controls = new OrbitControls(rig.camera, container);
  rig.controls.enableDamping = true;
  rig.controls.dampingFactor = 0.08;
  rig.controls.rotateSpeed = 0.6;

  for (const el of container.querySelectorAll(".view-canvas")) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    rig.views.push({ el, scene, points: null });
  }

  const resize = () => {
    const r = container.getBoundingClientRect();
    rig.renderer.setSize(r.width, r.height);
  };
  new ResizeObserver(resize).observe(container);
  resize();

  rig.renderer.setAnimationLoop(() => {
    rig.controls.update();
    const crect = container.getBoundingClientRect();
    rig.renderer.setScissorTest(false);
    rig.renderer.setClearColor(0x000000, 0);
    rig.renderer.clear();
    rig.renderer.setScissorTest(true);
    for (const v of rig.views) {
      const r = v.el.getBoundingClientRect();
      const left = r.left - crect.left, top = r.top - crect.top;
      const bottom = crect.height - top - r.height;
      rig.renderer.setViewport(left, bottom, r.width, r.height);
      rig.renderer.setScissor(left, bottom, r.width, r.height);
      rig.camera.aspect = r.width / r.height;
      rig.camera.updateProjectionMatrix();
      rig.renderer.render(v.scene, rig.camera);
    }
  });
}

function setGeometry() {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(state.positions, 3));
  const mat = () => new THREE.PointsMaterial({ size: 0.014, vertexColors: true, sizeAttenuation: true });
  for (const v of rig.views) {
    if (v.points) {
      v.scene.remove(v.points);
      v.points.geometry.dispose();
      v.points.material.dispose();
    }
    const g = geo.clone();
    v.points = new THREE.Points(g, mat());
    v.scene.add(v.points);
  }
  const f = state.manifest.frames.find((x) => x.id === state.frame);
  rig.camera.position.set(0, 0.05, 1.8);
  rig.camera.up.set(0, 1, 0);
  rig.controls.target.set(0, f.centroid[1], f.centroid[2]);
  rig.controls.update();
}

function setViewColors(key, colors) {
  const v = rig.views.find((x) => x.el.dataset.view === key);
  v.points.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  v.points.geometry.attributes.color.needsUpdate = true;
}

/* ---------- UI ---------- */

function chip(rgb, label) {
  return `<span class="chip"><i style="background:rgb(${rgb.join(",")})"></i>${label}</span>`;
}

function renderLegend() {
  const m = state.manifest;
  if (state.mode === "err") {
    $("#legend").innerHTML =
      chip(m.error_palette.correct, "correct") +
      chip(m.error_palette.wrong, "misclassified") +
      chip(m.error_palette.ignored, "ignored");
    return;
  }
  const f = m.frames.find((x) => x.id === state.frame);
  $("#legend").innerHTML = f.classes_present
    .map((c) => chip(m.class_palette[c], c.replaceAll("_", " ")))
    .join("");
}

async function updateModelViews() {
  const [pa, pb] = await Promise.all([loadPred(state.a), loadPred(state.b)]);
  const make = (p) => (state.mode === "err" ? errorColors(state.gt, p) : semanticColors(p));
  setViewColors("a", make(pa));
  setViewColors("b", make(pb));
  const suffix = state.mode === "err" ? "error map" : "prediction";
  $("#label-a").textContent = `${MODELS[state.a]} — ${suffix}`;
  $("#label-b").textContent = `${MODELS[state.b]} — ${suffix}`;
  renderLegend();
}

async function selectFrame(frameId) {
  $("#loading").style.display = "flex";
  state.frame = frameId;
  $("#sel-frame").value = frameId;
  await loadFrame(frameId);
  setGeometry();
  setViewColors("rgb", rgbColors(state.rgb));
  setViewColors("gt", semanticColors(state.gt));
  await updateModelViews();
  $("#loading").style.display = "none";
}

function fillSelects() {
  const groups = {};
  state.manifest.frames.forEach((f) => (groups[f.group] ??= []).push(f));
  $("#sel-frame").innerHTML = Object.entries(groups)
    .map(([g, fs]) =>
      `<optgroup label="${g}">${fs.map((f) => `<option value="${f.id}">${f.label}</option>`).join("")}</optgroup>`)
    .join("");
  const opts = (cur) => Object.entries(MODELS)
    .map(([k, n]) => `<option value="${k}" ${k === cur ? "selected" : ""}>${n}</option>`)
    .join("");
  $("#sel-a").innerHTML = opts(state.a);
  $("#sel-b").innerHTML = opts(state.b);
}

function feelingLucky() {
  const frames = state.manifest.frames;
  const keys = Object.keys(MODELS);
  const frame = frames[Math.floor(Math.random() * frames.length)].id;
  const a = keys[Math.floor(Math.random() * keys.length)];
  let b = a;
  while (b === a) b = keys[Math.floor(Math.random() * keys.length)];
  state.a = a; state.b = b;
  $("#sel-a").value = a;
  $("#sel-b").value = b;
  selectFrame(frame);
}

async function main() {
  const r = await fetch(`${HF}/metrics/compare_manifest.json`);
  state.manifest = await r.json();
  fillSelects();
  setupViews();

  $("#sel-frame").addEventListener("change", (e) => selectFrame(e.target.value));
  $("#sel-a").addEventListener("change", async (e) => { state.a = e.target.value; await updateModelViews(); });
  $("#sel-b").addEventListener("change", async (e) => { state.b = e.target.value; await updateModelViews(); });
  $("#sel-mode").addEventListener("change", async (e) => { state.mode = e.target.value; await updateModelViews(); });
  $("#lucky").addEventListener("click", feelingLucky);

  await selectFrame(state.manifest.frames[0].id);
}

main().catch((err) => {
  console.error(err);
  $("#loading").textContent = `Failed to load: ${err.message}`;
});
