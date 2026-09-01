# Bench3DS³ — Website

Static website for Bench3DS³ (Benchmarking 3D Semantic Segmentation Approaches in
Synthetic Industrial Automation Environments): nine models compared on
Isaac-Sim-generated RGB-D data. Published via GitHub Pages.

All metrics and media are fetched at runtime from the public Hugging Face dataset repo
[`min99ian/industrial-semseg-web`](https://huggingface.co/datasets/min99ian/industrial-semseg-web);
this repository contains only the HTML/CSS/JS shell.

## Local preview

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Structure

- `index.html` — single-page interactive prediction explorer
- `js/app.js` — data fetching and three.js four-view point cloud rendering
- `css/style.css` — styling
