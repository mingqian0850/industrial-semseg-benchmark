# Industrial 3D Semantic Segmentation Benchmark — Website

Static benchmark website for a synthetic industrial 3D semantic segmentation study
(nine models, Isaac-Sim-generated RGB-D data). Published via GitHub Pages.

All metrics and media are fetched at runtime from the public Hugging Face dataset repo
[`min99ian/industrial-semseg-web`](https://huggingface.co/datasets/min99ian/industrial-semseg-web);
this repository contains only the HTML/CSS/JS shell.

## Local preview

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Structure

- `index.html` — single-page layout (leaderboard, robustness, cross-robot, efficiency, dataset, downloads)
- `js/app.js` — data fetching and rendering (Chart.js from CDN)
- `css/style.css` — styling
