# From understanding scenes to controlling motion

A research blog exploring robot control with Astra and MolmoAct2: a six-task
suite, experimental results, interactive comparisons, and recorded demonstrations.

Website: https://llong-cs.github.io/llms-for-robot-control/

## Preview locally

This repository contains a complete static site, including videos, preview
images, and the recorded model outputs used by the page. No build step is needed.

```bash
python3 -m http.server 8000
```

Open http://localhost:8000/ in a browser.

## Publish updates

GitHub Pages publishes the root of the `gh-pages` branch. `.nojekyll` tells
GitHub to serve the existing static files directly.

After editing `main.html`, keep the home-page copy synchronized and push both
branches:

```bash
cp main.html index.html
git add .
git commit -m "Update research blog"
git push origin main
git push origin HEAD:gh-pages
```

The page uses relative links so it works both locally and under the repository's
GitHub Pages URL. Keep the media files and JSON manifests alongside the HTML,
CSS, and JavaScript when moving or updating the site.
