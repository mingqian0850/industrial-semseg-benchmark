# Data generation tools

Scripts that produce the Hugging Face assets the site consumes
([`min99ian/industrial-semseg-web`](https://huggingface.co/datasets/min99ian/industrial-semseg-web)).
They run on the machine that holds the datasets and model prediction outputs;
paths are configured at the top of each script.

- `export_compare_pointclouds.py` — exports one `media/pc/{sample}/frame.bin.gz`
  per test frame (quantized coords, RGB, GT, source-pixel uv, and the
  predictions of all nine models concatenated and gzipped) plus
  `metrics/compare_manifest.json` (schema v3). Re-runs skip existing packs.
- `export_frame_2d.py` — renders the static 2D reference images
  (`rgb2d.webp`, `gt2d.webp`) from the raw Isaac Sim frames, using the same
  hash palette as the viewer.

Upload after either script:

```bash
hf upload-large-folder min99ian/industrial-semseg-web ~/web_release/hf-web-pc --repo-type dataset
```
