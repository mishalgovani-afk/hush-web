#!/usr/bin/env bash
# Assemble a deployable static site in web/dist: app files + self-hosted MediaPipe runtime + face model.
set -euo pipefail
cd "$(dirname "$0")/.."
MP_VERSION=1.2.0
MODEL_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
MODEL_SHA256=64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

rm -rf dist && mkdir -p dist/vendor/tasks-vision/wasm dist/models
cp index.html styles.css app.js core.js sw.js manifest.webmanifest dist/
cp -r icons dist/

tmp=$(mktemp -d)
(cd "$tmp" && npm pack "@mediapipe/tasks-vision@${MP_VERSION}" --silent >/dev/null && tar xzf ./*.tgz)
cp "$tmp/package/vision_bundle.mjs" dist/vendor/tasks-vision/
# The plain (SIMD) and no-SIMD builds; the "module" variant is unused.
cp "$tmp"/package/wasm/vision_wasm_internal.{js,wasm} "$tmp"/package/wasm/vision_wasm_nosimd_internal.{js,wasm} dist/vendor/tasks-vision/wasm/
cp "$tmp/package/LICENSE" dist/vendor/tasks-vision/ 2>/dev/null || true
rm -rf "$tmp"

if [ -f ../windows/hushcam/models/face_landmarker.task ]; then
  cp ../windows/hushcam/models/face_landmarker.task dist/models/
else
  curl -sSfL -o dist/models/face_landmarker.task "$MODEL_URL"
fi
echo "${MODEL_SHA256}  dist/models/face_landmarker.task" | sha256sum -c -
touch dist/.nojekyll
du -sh dist
