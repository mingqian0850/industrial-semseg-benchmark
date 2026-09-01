/* Interactive four-view point cloud comparison over all test frames.
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
  split: null,     // split object from manifest
  frameIdx: 0,     // index within split.frames
  a: "volt",
  b: "ditr",
  mode: "pred",
  positions: null, rgb: null, gt: null, uv: null,
  preds: new Map(),
  loadToken: 0,
};

const $ = (s) => document.querySelector(s);

const currentFrame = () => state.split.frames[state.frameIdx];
const sampleId = () => `${state.split.dataset}__${currentFrame().f}`;

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
  const n = rgb.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = rgb[i] / 255;
  return out;
}

/* ---------- data loading ---------- */

async function fetchBin(sample, file) {
  const r = await fetch(`${HF}/media/pc/${sample}/${file}`);
  if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

async function loadFrameData() {
  const f = currentFrame();
  const sample = sampleId();
  const [coordBuf, rgbBuf, gtBuf, uvBuf] = await Promise.all([
    fetchBin(sample, "coord.bin"),
    fetchBin(sample, "rgb.bin"),
    fetchBin(sample, "gt.bin"),
    fetchBin(sample, "uv.bin"),
  ]);
  const q = new Uint16Array(coordBuf);
  const n = q.length / 3;
  const pos = new Float32Array(n * 3);
  const [mnx, mny, mnz, mxx, mxy, mxz] = f.b;
  const sx = (mxx - mnx) / 65535, sy = (mxy - mny) / 65535, sz = (mxz - mnz) / 65535;
  for (let i = 0; i < n; i++) {
    pos[i * 3] = mnx + q[i * 3] * sx;
    pos[i * 3 + 1] = mny + q[i * 3 + 1] * sy;
    pos[i * 3 + 2] = mnz + q[i * 3 + 2] * sz;
  }
  state.positions = pos;
  state.rgb = new Uint8Array(rgbBuf);
  state.gt = new Int8Array(gtBuf);
  state.uv = new Uint16Array(uvBuf);
  state.preds.clear();
}

async function loadPred(model) {
  if (!state.preds.has(model)) {
    const buf = await fetchBin(sampleId(), `${model}.bin`);
    state.preds.set(model, new Int8Array(buf));
  }
  return state.preds.get(model);
}

/* ---------- three.js four-view rig ---------- */

const rig = { renderer: null, camera: null, controls: null, views: [] };

function setupViews() {
  const container = $("#views");
  rig.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "high-performance" });
  // Pass vertex colors through unchanged so 3D views match the 2D reference
  // images and legend chips exactly (default sRGB conversion brightens them).
  rig.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
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
  const f = currentFrame();
  rig.camera.position.set(0, 0.05, 1.8);
  rig.camera.up.set(0, 1, 0);
  rig.controls.target.set(0, f.c[1], f.c[2]);
  rig.controls.update();
}

function setViewColors(key, colors) {
  const v = rig.views.find((x) => x.el.dataset.view === key);
  v.points.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  v.points.geometry.attributes.color.needsUpdate = true;
}

/* ---------- 2D pred / error map (points splatted back to source pixels) ---------- */

const W2D = 612, H2D = 512;
const PAL2D = [];
for (let i = -1; i < 23; i++) PAL2D[i + 1] = hashColor(i);
const ERR2D = { correct: [120, 120, 120], wrong: [229, 57, 53], ignored: [0, 0, 0] };

function render2d(canvas, labels, errMode) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W2D, H2D);
  const d = img.data;
  for (let p = 0; p < W2D * H2D; p++) {
    d[p * 4] = 15; d[p * 4 + 1] = 23; d[p * 4 + 2] = 42; d[p * 4 + 3] = 255;
  }
  const { uv, gt } = state;
  for (let i = 0; i < labels.length; i++) {
    const c = errMode
      ? (gt[i] < 0 ? ERR2D.ignored : gt[i] === labels[i] ? ERR2D.correct : ERR2D.wrong)
      : PAL2D[labels[i] + 1];
    const u = uv[i * 2], v = uv[i * 2 + 1];
    // 3x3 splat: 120k points cover ~40% of pixels, this fills most gaps
    for (let y = Math.max(v - 1, 0); y <= Math.min(v + 1, H2D - 1); y++) {
      for (let x = Math.max(u - 1, 0); x <= Math.min(u + 1, W2D - 1); x++) {
        const o = (y * W2D + x) * 4;
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
      }
    }
  }
  ctx.putImageData(img, 0, 0);
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
  $("#legend").innerHTML = currentFrame().cls
    .map((i) => chip(m.class_palette[m.classes[i]], m.classes[i].replaceAll("_", " ")))
    .join("");
}

async function updateModelViews() {
  const token = state.loadToken;
  const [pa, pb] = await Promise.all([loadPred(state.a), loadPred(state.b)]);
  if (token !== state.loadToken) return;
  const make = (p) => (state.mode === "err" ? errorColors(state.gt, p) : semanticColors(p));
  setViewColors("a", make(pa));
  setViewColors("b", make(pb));
  const err = state.mode === "err";
  const suffix = err ? "error map" : "prediction";
  $("#label-a").textContent = `${MODELS[state.a]} — ${suffix}`;
  $("#label-b").textContent = `${MODELS[state.b]} — ${suffix}`;
  render2d($("#cv2d-a"), pa, err);
  render2d($("#cv2d-b"), pb, err);
  $("#cap2d-a").textContent = `${MODELS[state.a]} — ${suffix} (2D)`;
  $("#cap2d-b").textContent = `${MODELS[state.b]} — ${suffix} (2D)`;
  renderLegend();
}

function updateRef2d() {
  // Noise variants perturb depth only, so they share the clean frame's 2D images.
  const sample = sampleId().replace(/_(mild|medium|strong)_noise__/, "__");
  $("#img2d-rgb").src = `${HF}/media/pc/${sample}/rgb2d.webp`;
  $("#img2d-gt").src = `${HF}/media/pc/${sample}/gt2d.webp`;
}

async function showFrame() {
  const token = ++state.loadToken;
  $("#loading").style.display = "flex";
  $("#sel-frame").value = String(state.frameIdx);
  updateRef2d();
  try {
    await loadFrameData();
    if (token !== state.loadToken) return;
    setGeometry();
    setViewColors("rgb", rgbColors(state.rgb));
    setViewColors("gt", semanticColors(state.gt));
    await updateModelViews();
  } finally {
    if (token === state.loadToken) $("#loading").style.display = "none";
  }
}

function fillFrameSelect() {
  $("#sel-frame").innerHTML = state.split.frames
    .map((f, i) => `<option value="${i}">${f.f.replace("frame_", "frame ")}</option>`)
    .join("");
}

function setSplit(splitId, frameIdx = 0) {
  state.split = state.manifest.splits.find((s) => s.id === splitId);
  state.frameIdx = frameIdx;
  $("#sel-split").value = splitId;
  fillFrameSelect();
  return showFrame();
}

function step(delta) {
  const n = state.split.frames.length;
  state.frameIdx = (state.frameIdx + delta + n) % n;
  showFrame();
}

function fillSelects() {
  $("#sel-split").innerHTML = state.manifest.splits
    .map((s) => `<option value="${s.id}">${s.label} (${s.frames.length})</option>`)
    .join("");
  const opts = (cur) => Object.entries(MODELS)
    .map(([k, n]) => `<option value="${k}" ${k === cur ? "selected" : ""}>${n}</option>`)
    .join("");
  $("#sel-a").innerHTML = opts(state.a);
  $("#sel-b").innerHTML = opts(state.b);
}

function feelingLucky() {
  const splits = state.manifest.splits;
  const sp = splits[Math.floor(Math.random() * splits.length)];
  const idx = Math.floor(Math.random() * sp.frames.length);
  const keys = Object.keys(MODELS);
  const a = keys[Math.floor(Math.random() * keys.length)];
  let b = a;
  while (b === a) b = keys[Math.floor(Math.random() * keys.length)];
  state.a = a; state.b = b;
  $("#sel-a").value = a;
  $("#sel-b").value = b;
  setSplit(sp.id, idx);
}

async function main() {
  const r = await fetch(`${HF}/metrics/compare_manifest.json`);
  state.manifest = await r.json();
  fillSelects();
  setupViews();

  $("#sel-split").addEventListener("change", (e) => setSplit(e.target.value));
  $("#sel-frame").addEventListener("change", (e) => { state.frameIdx = Number(e.target.value); showFrame(); });
  $("#prev").addEventListener("click", () => step(-1));
  $("#next").addEventListener("click", () => step(1));
  $("#sel-a").addEventListener("change", async (e) => { state.a = e.target.value; await updateModelViews(); });
  $("#sel-b").addEventListener("change", async (e) => { state.b = e.target.value; await updateModelViews(); });
  $("#sel-mode").addEventListener("change", async (e) => { state.mode = e.target.value; await updateModelViews(); });
  $("#lucky").addEventListener("click", feelingLucky);
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });

  await setSplit(state.manifest.splits[0].id);
}

main().catch((err) => {
  console.error(err);
  $("#loading").textContent = `Failed to load: ${err.message}`;
});
