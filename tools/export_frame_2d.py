#!/usr/bin/env python3
"""Export static 2D reference images (RGB + semantic GT) for every viewer frame.

Writes media/pc/{sample}/rgb2d.webp and gt2d.webp next to the existing binaries.
Colors match the 3D viewer (hash palette on train ids; unlabeled = dark gray).
Skips frames whose outputs already exist.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HOME = Path.home()
DATA = HOME / "isaac_sim_data_collector/data"
RAW = DATA / "isaac_sim_raw"
OUT = HOME / "web_release/hf-web-pc"

label_map = json.load(open(DATA / "ptv3_dataset/label_mapping.json"))
NAME_TO_TRAIN = label_map["label_to_train_id"]
IGNORE = set(label_map["ignore_labels"])


def hash_color(tid: int) -> tuple[int, int, int]:
    if tid < 0:
        return (40, 40, 40)
    if tid == 0:
        return (20, 20, 20)
    return ((tid * 37 + 17) % 255, (tid * 67 + 29) % 255, (tid * 97 + 43) % 255)


def export(sample_dir: Path) -> bool:
    sample = sample_dir.name
    dataset, frame = sample.split("__")
    src = RAW / dataset / "frames" / frame
    rgb_out = sample_dir / "rgb2d.webp"
    gt_out = sample_dir / "gt2d.webp"
    if rgb_out.exists() and gt_out.exists():
        return False

    Image.open(src / "rgb.png").convert("RGB").save(rgb_out, "WEBP", quality=87, method=4)

    seg = np.load(src / "semantic_segmentation.npy")
    info = json.load(open(src / "semantic_segmentation_info.json"))
    max_id = int(seg.max())
    lut = np.full((max_id + 1, 3), 40, dtype=np.uint8)
    for rid_str, meta in info["idToLabels"].items():
        rid = int(rid_str)
        if rid > max_id:
            continue
        name = meta.get("class", "")
        tid = -1 if name in IGNORE else NAME_TO_TRAIN.get(name, -1)
        lut[rid] = hash_color(tid)
    Image.fromarray(lut[seg]).save(gt_out, "WEBP", quality=90, method=4)
    return True


def main() -> None:
    dirs = sorted(p for p in (OUT / "media/pc").iterdir() if p.is_dir())
    done = 0
    for i, d in enumerate(dirs, 1):
        if export(d):
            done += 1
        if i % 200 == 0:
            print(f"{i}/{len(dirs)}", flush=True)
    print(f"done: {done} frames exported, {len(dirs)} total")


if __name__ == "__main__":
    main()
