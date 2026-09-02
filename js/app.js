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
  preds: null,     // Map model -> Int8Array, filled from the frame pack
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

/* ---------- data loading ----------
 * One gzip pack per frame holds coord/rgb/gt/uv plus the predictions of ALL
 * models (labels compress to ~7 KB each), so switching models needs no fetch.
 * Layout (little-endian, n points): coord u16*3 | rgb u8*3 | gt i8 | uv u16*2
 * | one i8 array per model in manifest.models order. */

async function fetchPack(sample, retry = 1) {
  try {
    const r = await fetch(`${HF}/media/pc/${sample}/frame.bin.gz`);
    if (!r.ok) throw new Error(`${sample}: HTTP ${r.status}`);
    const ds = new DecompressionStream("gzip");
    return await new Response(r.body.pipeThrough(ds)).arrayBuffer();
  } catch (e) {
    if (retry <= 0) throw e;
    await new Promise((res) => setTimeout(res, 800));
    return fetchPack(sample, retry - 1);
  }
}

/* Small LRU of decompressed packs; also used to prefetch the next frame. */
const packCache = new Map();

function getPack(sample) {
  if (!packCache.has(sample)) {
    if (packCache.size >= 8) packCache.delete(packCache.keys().next().value);
    const p = fetchPack(sample).catch((e) => { packCache.delete(sample); throw e; });
    packCache.set(sample, p);
  }
  return packCache.get(sample);
}

async function loadFrameData() {
  const f = currentFrame();
  const buf = await getPack(sampleId());
  const n = f.n;
  const q = new Uint16Array(buf, 0, n * 3);
  const pos = new Float32Array(n * 3);
  const [mnx, mny, mnz, mxx, mxy, mxz] = f.b;
  const sx = (mxx - mnx) / 65535, sy = (mxy - mny) / 65535, sz = (mxz - mnz) / 65535;
  for (let i = 0; i < n; i++) {
    pos[i * 3] = mnx + q[i * 3] * sx;
    pos[i * 3 + 1] = mny + q[i * 3 + 1] * sy;
    pos[i * 3 + 2] = mnz + q[i * 3 + 2] * sz;
  }
  state.positions = pos;
  state.rgb = new Uint8Array(buf, n * 6, n * 3);
  state.gt = new Int8Array(buf, n * 9, n);
  state.uv = new Uint16Array(buf, n * 10, n * 2);
  state.preds = new Map(
    state.manifest.models.map((m, i) => [m, new Int8Array(buf, n * 14 + i * n, n)]));
}

function prefetchNextFrame() {
  const frames = state.split.frames;
  const next = frames[(state.frameIdx + 1) % frames.length];
  getPack(`${state.split.dataset}__${next.f}`).catch(() => {});
}

/* ---------- three.js four-view rig ---------- */

const FULL_PR = Math.min(window.devicePixelRatio, 2);

const rig = {
  renderer: null, camera: null, controls: null, views: [],
  needsRender: true, lastSplit: null,
  w: 0, h: 0, pr: FULL_PR, lastInteract: 0,
};

function setupViews() {
  const container = $("#views");
  rig.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "high-performance" });
  // Pass vertex colors through unchanged so 3D views match the 2D reference
  // images and legend chips exactly (default sRGB conversion brightens them).
  rig.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  rig.renderer.setPixelRatio(rig.pr);
  rig.renderer.domElement.className = "views-gl";
  container.prepend(rig.renderer.domElement);

  rig.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
  rig.controls = new OrbitControls(rig.camera, container);
  rig.controls.enableDamping = true;
  rig.controls.dampingFactor = 0.08;
  rig.controls.rotateSpeed = 0.6;
  rig.controls.addEventListener("change", () => {
    rig.lastInteract = performance.now();
    rig.needsRender = true;
  });

  for (const el of container.querySelectorAll(".view-canvas")) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    rig.views.push({ el, scene, points: null, rect: null });
  }

  // Layout queries are cached here; the render loop never touches layout.
  const resize = () => {
    const cr = container.getBoundingClientRect();
    rig.w = cr.width; rig.h = cr.height;
    rig.renderer.setSize(rig.w, rig.h);
    for (const v of rig.views) {
      const r = v.el.getBoundingClientRect();
      v.rect = {
        left: r.left - cr.left,
        bottom: cr.height - (r.top - cr.top) - r.height,
        w: r.width, h: r.height,
      };
    }
    rig.needsRender = true;
  };
  new ResizeObserver(resize).observe(container);
  resize();

  // Render only when the camera moved or data changed; an idle page costs no
  // GPU. While interacting, drop to 1x pixel ratio (4x less fill on hidpi) and
  // restore full resolution 250 ms after the last camera change.
  rig.renderer.setAnimationLoop(() => {
    const moved = rig.controls.update();
    const interacting = performance.now() - rig.lastInteract < 250;
    const targetPR = interacting ? 1 : FULL_PR;
    if (targetPR !== rig.pr) {
      rig.pr = targetPR;
      rig.renderer.setPixelRatio(targetPR);
      rig.renderer.setSize(rig.w, rig.h, false);
      rig.needsRender = true;
    }
    if (!moved && !rig.needsRender) return;
    rig.needsRender = false;
    rig.renderer.setScissorTest(false);
    rig.renderer.setClearColor(0x000000, 0);
    rig.renderer.clear();
    rig.renderer.setScissorTest(true);
    for (const v of rig.views) {
      const { left, bottom, w, h } = v.rect;
      rig.renderer.setViewport(left, bottom, w, h);
      rig.renderer.setScissor(left, bottom, w, h);
      rig.camera.aspect = w / h;
      rig.camera.updateProjectionMatrix();
      rig.renderer.render(v.scene, rig.camera);
    }
  });
}

function makePointsMaterial() {
  const m = new THREE.PointsMaterial({ size: 0.014, vertexColors: true, sizeAttenuation: true });
  // Cap the on-screen point size: when zoomed in close, unbounded attenuated
  // points blow up fill rate (120k points x 4 views) and zooming stutters.
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader.replace(
      "if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );",
      "if ( isPerspective ) gl_PointSize = min( gl_PointSize * ( scale / - mvPosition.z ), 16.0 );",
    );
  };
  return m;
}

function setGeometry() {
  // One position attribute shared by all four views (one GPU buffer).
  const posAttr = new THREE.BufferAttribute(state.positions, 3);
  for (const v of rig.views) {
    if (v.points) {
      v.scene.remove(v.points);
      v.points.geometry.dispose();
      v.points.material.dispose();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", posAttr);
    v.points = new THREE.Points(g, makePointsMaterial());
    v.scene.add(v.points);
  }
  // Keep the user's viewpoint when stepping through frames of the same split;
  // reset only when the split (scene/robot) changes.
  if (rig.lastSplit !== state.split.id) {
    rig.lastSplit = state.split.id;
    const f = currentFrame();
    rig.camera.position.set(0, 0.05, 1.8);
    rig.camera.up.set(0, 1, 0);
    rig.controls.target.set(0, f.c[1], f.c[2]);
    rig.controls.update();
  }
  rig.needsRender = true;
}

function setViewColors(key, colors) {
  const v = rig.views.find((x) => x.el.dataset.view === key);
  v.points.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  rig.needsRender = true;
}

/* ---------- 2D pred / error map (points splatted back to source pixels) ---------- */

const W2D = 612, H2D = 512;
// Packed ABGR (little-endian RGBA bytes) so each pixel is a single u32 write.
const px32 = ([r, g, b]) => (0xff000000 | (b << 16) | (g << 8) | r) >>> 0;
const BG32 = px32([15, 23, 42]);
const PAL32 = new Uint32Array(24);
for (let i = -1; i < 23; i++) PAL32[i + 1] = px32(hashColor(i));
const ERR32 = { correct: px32([120, 120, 120]), wrong: px32([229, 57, 53]), ignored: px32([0, 0, 0]) };

function render2d(canvas, labels, errMode) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W2D, H2D);
  const px = new Uint32Array(img.data.buffer);
  px.fill(BG32);
  const { uv, gt } = state;
  for (let i = 0; i < labels.length; i++) {
    const c = errMode
      ? (gt[i] < 0 ? ERR32.ignored : gt[i] === labels[i] ? ERR32.correct : ERR32.wrong)
      : PAL32[labels[i] + 1];
    const u = uv[i * 2], v = uv[i * 2 + 1];
    // 3x3 splat: 120k points cover ~40% of pixels, this fills most gaps
    for (let y = Math.max(v - 1, 0); y <= Math.min(v + 1, H2D - 1); y++) {
      for (let x = Math.max(u - 1, 0); x <= Math.min(u + 1, W2D - 1); x++) {
        px[y * W2D + x] = c;
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

function updateModelViews() {
  if (!state.preds) return; // first frame still loading
  const pa = state.preds.get(state.a);
  const pb = state.preds.get(state.b);
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
    updateModelViews();
    prefetchNextFrame();
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
  $("#sel-a").addEventListener("change", (e) => { state.a = e.target.value; updateModelViews(); });
  $("#sel-b").addEventListener("change", (e) => { state.b = e.target.value; updateModelViews(); });
  $("#sel-mode").addEventListener("change", (e) => { state.mode = e.target.value; updateModelViews(); });
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
