#!/usr/bin/env python3
"""Export per-frame point cloud binaries for the three.js viewer — ALL test frames.

Covers the clean test split (1012 frames, 3 scenes), the three noise levels
(mild/medium/strong, 1012 frames each), and all 7 robot generalization splits
(~800 frames). One gzip pack per frame at media/pc/{sample}/frame.bin.gz,
concatenating (little-endian, n = point count from the manifest):
  coord   uint16 x3 per point, quantized to the frame bbox
  rgb     uint8 x3 per point
  gt      int8 per point (-1 ignored)
  uv      uint16 x2 per point, source pixel (u, v) from the aligned DiTR set
  preds   int8 per point per model, in MODEL_ORDER

Labels compress to almost nothing, so shipping all nine model predictions in
the pack costs ~60 KB while making model switching in the viewer free. gzip
uses mtime=0 so re-runs are byte-identical (upload dedupe).

All 120k points are exported (no cropping: synthetic coords are bounded within
~10 m, so min/max quantization is lossless enough and cropping used to cut
visible chunks such as the robot top, the closest region to the camera).

Writes metrics/compare_manifest.json (compact schema v3, grouped by split).
Re-runs skip frames whose frame.bin.gz exists (written via tmp + rename).
"""
import gzip
import json
from pathlib import Path

import numpy as np

HOME = Path.home()
DATA = HOME / "isaac_sim_data_collector/data"
OUT = HOME / "web_release/hf-web-pc"

PCEPT = HOME / "ws_Pointcept/exp/industrial"
DITR = HOME / "ws_DITR/ditr/exp/industrial"
VOLT = HOME / "ws_volt/Volt/exp/industrial"

ROBOTS = ["ur30", "ur10", "tm12", "panda", "nex10", "kawasaki", "kuka"]
ROBOT_LABELS = {
    "ur30": "UR30", "ur10": "UR10", "tm12": "TM12", "panda": "Franka Panda",
    "nex10": "Yaskawa NEX10", "kawasaki": "Kawasaki", "kuka": "KUKA KR210",
}
NOISE_LEVELS = ["mild", "medium", "strong"]
MODEL_ORDER = ["ptv3", "litept", "oacnn", "ptv3cac", "octformer",
               "sonatalin", "sonataft", "ditr", "volt"]


def exp_dirs(condition: str) -> dict[str, Path]:
    if condition == "clean":
        return {
            "ptv3": PCEPT / "ptv3_base_23cls_gpu1_test",
            "litept": PCEPT / "litept_small_23cls_gpu1",
            "oacnn": PCEPT / "oacnn_base_23cls_gpu1_run_2",
            "ptv3cac": PCEPT / "semseg-cac-v1m1-0-base",
            "octformer": PCEPT / "octformer_base_23cls_gpu1",
            "sonatalin": PCEPT / "sonata_industrial_lin",
            "sonataft": PCEPT / "sonata_industrial_ft_test",
            "ditr": DITR / "injection_aligned_23cls_gpu1",
            "volt": VOLT / "semseg-volt-small-ft",
        }
    if condition in NOISE_LEVELS:
        n = condition
        return {
            "ptv3": PCEPT / f"ptv3_base_23cls_gpu1_test_{n}_noise",
            "litept": PCEPT / f"litept_small_23cls_gpu1_test_{n}_noise",
            "oacnn": PCEPT / f"oacnn_base_23cls_gpu1_test_{n}_noise",
            "ptv3cac": PCEPT / f"semseg-cac-v1m1-0-base_test_{n}_noise",
            "octformer": PCEPT / f"octformer_base_23cls_gpu1_test_{n}_noise",
            "sonatalin": PCEPT / f"sonata_industrial_lin_test_{n}_noise",
            "sonataft": PCEPT / f"sonata_industrial_ft_test_{n}_noise",
            "ditr": DITR / f"injection_aligned_23cls_gpu1_test_{n}_noise",
            "volt": VOLT / f"semseg-volt-small-ft-test-{n}-noise",
        }
    r = condition
    return {
        "ptv3": PCEPT / f"ptv3_base_23cls_gpu1_test_robot_{r}",
        "litept": PCEPT / f"litept_small_23cls_gpu1_test_robot_{r}",
        "oacnn": PCEPT / f"oacnn_base_23cls_gpu1_test_robot_{r}",
        "ptv3cac": PCEPT / f"semseg-cac-v1m1-0-base_test_robot_{r}",
        "octformer": PCEPT / f"octformer_base_23cls_gpu1_test_robot_{r}",
        "sonatalin": PCEPT / f"sonata_industrial_lin_test_robot_{r}",
        "sonataft": PCEPT / f"sonata_industrial_ft_test_robot_{r}",
        "ditr": DITR / f"injection_aligned_23cls_gpu1_test_robot_{r}",
        "volt": VOLT / f"semseg-volt-small-ft-test-robot-{r}",
    }


def scene_groups(root: Path) -> dict[str, list[str]]:
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    per_scene = {}
    for n in names:
        per_scene.setdefault(n.split("_layout")[0], []).append(n)
    return per_scene


def build_splits() -> list[dict]:
    """Each split: {id, label, dataset_root, exp, samples}."""
    splits = []
    clean_root = DATA / "ptv3_dataset/test"
    for scene, ns in sorted(scene_groups(clean_root).items()):
        splits.append({
            "id": f"clean_{scene}",
            "label": f"Clean test · {scene}",
            "root": clean_root,
            "exp": exp_dirs("clean"),
            "samples": ns,
        })
    for lvl in NOISE_LEVELS:
        root = DATA / f"ptv3_dataset_test_{lvl}_noise/test"
        for scene, ns in sorted(scene_groups(root).items()):
            splits.append({
                "id": f"{lvl}_{scene}",
                "label": f"{lvl.capitalize()} noise · {scene}",
                "root": root,
                "exp": exp_dirs(lvl),
                "samples": ns,
            })
    for r in ROBOTS:
        root = DATA / f"ptv3_dataset_test_robot_{r}/test"
        ns = sorted(p.name for p in root.iterdir() if p.is_dir())
        splits.append({
            "id": f"robot_{r}",
            "label": f"Robot generalization · {ROBOT_LABELS[r]}",
            "root": root,
            "exp": exp_dirs(r),
            "samples": ns,
        })
    return splits


def ditr_root(root: Path) -> Path:
    """ptv3 dataset root -> aligned DiTR dataset root (same samples/point order)."""
    name = root.parent.name.replace("ptv3_dataset", "ditr_injection_dataset_aligned")
    return root.parent.parent / name / root.name


def export_frame(root: Path, exp: dict[str, Path], sample: str) -> dict | None:
    # skip frames where any model's prediction is missing (a few skipped test frames)
    missing = [m for m, e in exp.items() if not (e / "result" / f"{sample}_pred.npy").is_file()]
    if missing:
        print(f"SKIP {sample}: missing preds for {missing}", flush=True)
        return None
    fdir = OUT / "media/pc" / sample
    sdir = root / sample
    coord = np.load(sdir / "coord.npy").astype(np.float32)
    gt = np.load(sdir / "segment.npy").reshape(-1).astype(np.int8)
    n = len(coord)

    bmin = coord.min(axis=0)
    bmax = coord.max(axis=0)
    centroid = coord.mean(axis=0)
    present = sorted(int(i) for i in np.unique(gt) if i >= 0)

    pack = fdir / "frame.bin.gz"
    if not pack.exists():
        fdir.mkdir(parents=True, exist_ok=True)
        scale = np.where(bmax > bmin, bmax - bmin, 1.0)
        q = np.round((coord - bmin) / scale * 65535.0).astype(np.uint16)
        rgb = np.load(sdir / "color.npy").astype(np.uint8)
        uv = np.load(ditr_root(root) / sample / "point2pixel.npy").reshape(-1, 2)
        parts = [q.tobytes(), rgb.tobytes(), gt.tobytes(),
                 np.round(uv).astype(np.uint16).tobytes()]
        for model in MODEL_ORDER:
            pred = np.load(exp[model] / "result" / f"{sample}_pred.npy").reshape(-1).astype(np.int8)
            assert len(pred) == n, f"{model} {sample}: {len(pred)} vs {n}"
            parts.append(pred.tobytes())
        tmp = fdir / "frame.bin.gz.tmp"
        tmp.write_bytes(gzip.compress(b"".join(parts), 6, mtime=0))
        tmp.rename(pack)
        for old in fdir.glob("*.bin"):
            old.unlink()

    return {
        "f": sample.split("__")[1],
        "n": int(len(coord)),
        "b": [round(float(v), 4) for v in bmin] + [round(float(v), 4) for v in bmax],
        "c": [round(float(v), 3) for v in centroid],
        "cls": present,
    }


def main() -> None:
    assert list(exp_dirs("clean")) == MODEL_ORDER
    OUT.mkdir(parents=True, exist_ok=True)
    label_map = json.load(open(DATA / "ptv3_dataset/label_mapping.json"))
    class_names = label_map["class_names"]

    manifest_splits = []
    total = 0
    for sp in build_splits():
        dataset = sp["samples"][0].split("__")[0]
        frames = []
        for sample in sp["samples"]:
            entry = export_frame(sp["root"], sp["exp"], sample)
            if entry is None:
                continue
            frames.append(entry)
            total += 1
            if total % 100 == 0:
                print(f"{total} frames done ({sp['id']})", flush=True)
        manifest_splits.append({
            "id": sp["id"],
            "label": sp["label"],
            "dataset": dataset,
            "frames": frames,
        })
        print(f"split {sp['id']}: {len(frames)} frames", flush=True)

    palette = {
        name: [int((i * 37 + 17) % 255), int((i * 67 + 29) % 255), int((i * 97 + 43) % 255)]
        if i > 0 else [20, 20, 20]
        for i, name in enumerate(class_names)
    }
    manifest = {
        "version": 3,
        "note": "frame.bin.gz = gzip of concat(coord uint16*3 quantized to b=[min,max], "
                "rgb uint8*3, gt int8, uv uint16*2 source pixels, then one int8 pred "
                "array per model in `models` order); n points per frame entry; "
                "sample path = media/pc/{dataset}__{f}/",
        "models": MODEL_ORDER,
        "classes": class_names,
        "class_palette": palette,
        "error_palette": {"correct": [120, 120, 120], "wrong": [229, 57, 53], "ignored": [0, 0, 0]},
        "splits": manifest_splits,
    }
    (OUT / "metrics").mkdir(exist_ok=True)
    (OUT / "metrics/compare_manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n")

    n = sum(1 for f in OUT.rglob("*") if f.is_file())
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e9
    print(f"\ndone: {OUT} ({n} files, {size:.2f} GB, {total} frames)")


if __name__ == "__main__":
    main()
