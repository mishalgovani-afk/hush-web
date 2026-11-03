// Offline support: cache the app shell on install, everything else (MediaPipe WASM, model) on first use.
const VERSION = "hush-v1";
const SHELL = ["./", "index.html", "styles.css", "app.js", "core.js", "manifest.webmanifest", "icons/icon.svg"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  const heavy = /\/(vendor|models)\//.test(req.url);
  // Heavy immutable assets: cache-first. App code: network-first so updates land, cache when offline.
  e.respondWith(heavy
    ? caches.match(req).then((hit) => hit || fetch(req).then((res) => { if (res.ok) caches.open(VERSION).then((c) => c.put(req, res.clone())); return res; }))
    : fetch(req).then((res) => { if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); } return res; })
        .catch(() => caches.match(req).then((hit) => hit || caches.match("index.html"))));
});
