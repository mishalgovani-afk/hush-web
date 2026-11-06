# Hush Web: silent speech in your phone's browser

Hush Web runs on **iPhone (Safari)**, Android (Chrome) and desktop browsers. Prop up your phone and silently mouth words at the front camera. Hush Web learns *your* lips and turns them into text. Everything runs on the device: MediaPipe's face model runs in WebAssembly, and no video or data leaves the phone.

It's the web version of HushCam (Windows) and shares its algorithm and dataset format. You can export training data from one and import it into the other.

> Source of truth: this is the web app from the Hush project (Mac / Windows / web).

> Why not AirPods? Phone browsers, like Windows, have no access to AirPods head motion. That API only exists in native Apple apps. So the web version reads lips through the camera.

## Open it on your iPhone

The page needs **HTTPS**, because browsers only allow camera access on secure pages. Pick one:

1. **GitHub Pages**: Settings → Pages → Source: **GitHub Actions**. Every push to `main` deploys to `https://mishalgovani-afk.github.io/hush-web/`.
2. **Any static host:** download the **Hush-Web** artifact from the latest CI run. Upload the folder to Netlify Drop, Cloudflare Pages, Vercel or similar.
3. **Locally:** run `./scripts/build.sh && npx serve dist`. Then open the https tunnel URL (for example `npx localtunnel --port 3000`) on your phone.

Then, in Safari, tap **Share → Add to Home Screen**. Hush opens full-screen like an app and works offline after the first load.

## Use it

| Tab | What you do |
|---|---|
| **Train** | A word is shown. Press and hold the big button, mouth the word, release. Hush moves on to the least-recorded word by itself. Record 8–15 takes per word, plus `_noise` takes (smiles, yawns, swallowing) that it learns to ignore. |
| **Check** | Leave-one-out accuracy vs 500 shuffled-label runs, with a p-value, histogram and confusion matrix. *Compare features* shows which lip measurements carry the signal. |
| **Decode** | Hold the button and mouth a word, or turn on **hands-free**: calibrate with your lips closed for 2 s, then just mouth words. Copy or share the transcript, or have words **spoken aloud**. |
| **Feed** | A running feed of @realDonaldTrump's Truth Social posts, newest first, refreshed every minute while open. New posts are highlighted, and a badge on the tab counts unseen ones. |
| **Data** | Export/import training data (compatible with Windows HushCam), switch camera, delete takes. |

The condition picker at the top (Silent mouthing / Whisper / Normal speech) is stored with every take, so you can compare conditions in Check.

## Develop

```sh
node --test tests/*.test.mjs   # core tests, including exact numeric parity with windows/hushcam/core.py
./scripts/build.sh             # → dist/ with self-hosted MediaPipe runtime + face model (checksum verified)
```

`core.js` has no DOM dependencies and runs in the browser and in Node. `app.js` handles the camera, UI and storage (IndexedDB). `window.__hush` exposes hooks that let automated tests feed lip frames without a real face.

## Truth Social feed

Truth Social doesn't let other websites read its posts from the browser. `.github/workflows/feed.yml` runs every 5 minutes (GitHub's minimum; runs can start a few minutes late). `scripts/fetch-feed.mjs` queries public sources (Truth Social's API, CNN's Truth Social archive, trumpstruth.org) and merges what they return. It converts posts to plain text and force-pushes a single `feed.json` to the `feed-data` branch. The Feed tab reads that file from raw.githubusercontent.com. Post text is always shown as plain text; links open the original post on Truth Social.

GitHub pauses scheduled workflows in repos with no activity for 60 days. If the feed stops updating, re-enable it under **Actions → Refresh Truth Social feed**.
