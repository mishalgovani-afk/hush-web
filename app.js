import * as core from "./core.js";

const MP_VERSION = "1.2.0";
const LOCAL = { bundle: "./vendor/tasks-vision/vision_bundle.mjs", wasm: "./vendor/tasks-vision/wasm", model: "./models/face_landmarker.task" };
const CDN = {
  bundle: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`,
  wasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
  model: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
};

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const pct = (x) => `${Math.round(x * 100)}%`;

// ------------------------------------------------------------------------------------------ settings

const DEFAULTS = {
  condition: "mouthed", groups: [...core.DEFAULT_GROUPS], deltas: false, smoothing: 3, band: 0.2, k: 1,
  maxDistance: 1.0, minMargin: 0.08, sensitivity: 4, permutations: 500, evalCondition: "all",
  autoAdvance: true, autoDetect: false, speak: false, facing: "user", mirror: true, tab: "train",
};
const settings = (() => {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("hush.settings") || "{}") }; } catch { return { ...DEFAULTS }; }
})();
function saveSettings() {
  classifier = null;
  segmenter.cfg.sensitivity = settings.sensitivity;
  try { localStorage.setItem("hush.settings", JSON.stringify(settings)); } catch { /* private mode */ }
}
const featureCfg = () => core.featureConfig({ groups: core.DEFAULT_GROUPS.filter((g) => settings.groups.includes(g)), deltas: settings.deltas, smoothing: settings.smoothing });
const classifierCfg = () => core.classifierConfig({ band: settings.band, k: settings.k, maxDistance: settings.maxDistance, minMargin: settings.minMargin });

// ------------------------------------------------------------------------------------------ storage (IndexedDB)

const store = {
  db: null,
  async open() {
    if (this.db) return this.db;
    this.db = await new Promise((res, rej) => {
      const r = indexedDB.open("hush", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("kv");
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
    return this.db;
  },
  async get(key) {
    const db = await this.open();
    return new Promise((res, rej) => { const r = db.transaction("kv").objectStore("kv").get(key); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  },
  async set(key, value) {
    const db = await this.open();
    return new Promise((res, rej) => { const t = db.transaction("kv", "readwrite"); t.objectStore("kv").put(value, key); t.oncomplete = res; t.onerror = () => rej(t.error); });
  },
};

let dataset = core.newDataset();
let saveTimer = null;
function saveDataset() {
  classifier = null;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => store.set("dataset", core.datasetToJSON(dataset)).catch((e) => toast(`Could not save: ${e.message}`)), 150);
}

// ------------------------------------------------------------------------------------------ state

let landmarker = null, stream = null, running = false, lastTs = 0, faceSeen = 0, fps = 0, lastFrameAt = 0;
let capturing = false, capture = [], captureMissed = 0, lastTake = null;
let selectedWord = null, classifier = null, evaluating = false;
const segmenter = new core.Segmenter(core.segmenterConfig({ sensitivity: settings.sensitivity }));
const transcriptWords = [];
const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");

// ------------------------------------------------------------------------------------------ camera + MediaPipe

async function loadLandmarker(status) {
  let vision, wasm, model;
  try {
    vision = await import(LOCAL.bundle); wasm = LOCAL.wasm; model = LOCAL.model;
    const head = await fetch(model, { method: "HEAD" });
    if (!head.ok) throw new Error("no local model");
  } catch {
    vision = await import(CDN.bundle); wasm = CDN.wasm; model = CDN.model;
  }
  status("Loading face model…");
  const files = await vision.FilesetResolver.forVisionTasks(wasm);
  const make = (delegate) => vision.FaceLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: model, delegate },
    runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { return await make("GPU"); } catch (e) { console.warn("GPU delegate failed, using CPU", e); return make("CPU"); }
}

async function startCamera() {
  const status = (t) => { $("startStatus").textContent = t; };
  $("startBtn").disabled = true;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw Object.assign(new Error("Camera needs a secure (https://) page."), { name: "Insecure" });
    }
    status("Asking for camera access…");
    await openStream();
    if (!landmarker) { status("Loading lip tracker…"); landmarker = await loadLandmarker(status); }
    $("startPanel").hidden = true;
    $("holdBtn").disabled = false;
    running = true;
    scheduleFrame();
  } catch (e) {
    console.error(e);
    const msg = e.name === "NotAllowedError"
      ? "Camera access was blocked. On iPhone: Settings → Safari → Camera → Allow, then reload."
      : e.name === "NotFoundError" ? "No camera found on this device." : e.message || String(e);
    status(msg);
    $("startBtn").disabled = false;
    $("startBtn").textContent = "Try again";
  }
}

async function openStream() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: settings.facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
  });
  video.srcObject = stream;
  await video.play();
}

function scheduleFrame() {
  if (!running) return;
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onVideoFrame);
  else requestAnimationFrame(onVideoFrame);
}

function onVideoFrame() {
  try {
    if (video.readyState >= 2 && landmarker) processVideoFrame();
  } catch (e) { console.error(e); }
  scheduleFrame();
}

function processVideoFrame() {
  const now = performance.now();
  const ts = Math.max(now, lastTs + 1);
  lastTs = ts;
  if (lastFrameAt) fps = fps ? 0.9 * fps + 0.1 * (1000 / (now - lastFrameAt)) : 1000 / (now - lastFrameAt);
  lastFrameAt = now;
  const res = landmarker.detectForVideo(video, ts);
  const w = video.videoWidth, h = video.videoHeight;
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  ctx.clearRect(0, 0, w, h);
  if (!res.faceLandmarks?.length) { handleFrame(null); return; }
  const pts = res.faceLandmarks[0].map((p) => [p.x * w, p.y * h]);
  const blend = {};
  for (const c of res.faceBlendshapes?.[0]?.categories ?? []) blend[c.categoryName] = c.score;
  drawLips(pts, w);
  handleFrame(core.lipFrameFromLandmarks(pts, blend, ts / 1000));
}

function drawLips(pts, w) {
  ctx.lineWidth = Math.max(1.5, w / 320);
  ctx.lineJoin = "round";
  for (const [loop, color] of [[core.LIP_OUTLINE, "#34c77b"], [core.LIP_INNER, "#ff9f43"]]) {
    ctx.beginPath();
    loop.forEach((i, k) => (k ? ctx.lineTo(...pts[i]) : ctx.moveTo(...pts[i])));
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.stroke();
  }
}

// ------------------------------------------------------------------------------------------ frame routing

function handleFrame(f) {
  if (!f) { if (capturing) captureMissed++; updateLive(); return; }
  faceSeen = performance.now();
  if (capturing) {
    capture.push(f);
    if (capture.length > 300) endCapture();
  } else if (settings.autoDetect && settings.tab === "decode") {
    const seg = segmenter.process(f);
    if (seg) decode(seg, "hands-free");
  }
  updateLive();
}

function updateLive() {
  const ok = performance.now() - faceSeen < 500;
  const chip = $("faceChip");
  chip.className = `chip ${running ? (ok ? "ok" : "lost") : ""}`;
  $("faceText").textContent = !running ? "Camera off" : ok ? `Lips tracked · ${Math.round(fps)} fps` : "Show your face";
  $("recChip").hidden = !capturing;
  $("recCount").textContent = capture.length;
  if (settings.tab === "decode" && settings.autoDetect) renderSegmenter();
}

// ------------------------------------------------------------------------------------------ capture

function beginCapture() {
  if (capturing || !running) return;
  if (settings.tab === "train" && !selectedWord) { toast("Add a word first."); return; }
  if (settings.tab !== "train" && settings.tab !== "decode") showTab("decode");
  capturing = true;
  capture = [];
  captureMissed = 0;
  $("holdBtn").classList.add("active");
  $("holdText").textContent = "Release when done";
  document.querySelector(".prompt").classList.toggle("capturing", settings.tab === "train");
  updateLive();
}

function endCapture() {
  if (!capturing) return;
  capturing = false;
  const frames = capture, missed = captureMissed;
  capture = [];
  $("holdBtn").classList.remove("active");
  $("holdText").textContent = "Hold to capture";
  document.querySelector(".prompt").classList.remove("capturing");
  updateLive();
  if (frames.length < 6) { toast(missed ? "Couldn't see your lips. Face the camera." : "Too short. Hold for the whole word."); return; }
  if (settings.tab === "train") addTake(frames, missed);
  else decode(frames, "hold");
}

function addTake(frames, missed) {
  const take = { id: core.newId(), label: selectedWord, condition: settings.condition, created: new Date().toISOString(), frames };
  dataset.takes.push(take);
  lastTake = take;
  saveDataset();
  const n = core.countTakes(dataset, take.label, settings.condition);
  $("takeInfo").textContent = `Saved take ${n} of “${take.label}” · ${frames.length} frames${missed ? ` · ${missed} without a face` : ""}`;
  if (settings.autoAdvance) selectedWord = core.leastRecorded(dataset, settings.condition, take.label);
  renderTrain();
}

// ------------------------------------------------------------------------------------------ decode

function trainingTakes() {
  return core.filteredTakes(dataset, settings.evalCondition === "all" ? null : settings.evalCondition);
}

function getClassifier() {
  if (!classifier) {
    const cfg = featureCfg();
    classifier = new core.TemplateClassifier(trainingTakes().map((t) => [t.label, core.extract(t.frames, cfg)]), classifierCfg());
  }
  return classifier;
}

function decode(frames, source) {
  if (!settings.groups.length) return;
  const p = getClassifier().classify(core.extract(frames, featureCfg()));
  const word = $("decodedWord");
  word.textContent = p.label ?? "—";
  word.classList.toggle("rejected", !p.label);
  $("decodedDetail").textContent = p.rejection ?? `${pct(p.margin)} margin · ${frames.length} frames · ${source}`;
  $("ranked").replaceChildren(...p.ranked.slice(0, 6).map(([l, d]) => el("div", {}, el("span", { textContent: l }), el("span", { textContent: d.toFixed(3) }))));
  if (p.label) {
    transcriptWords.push(p.label);
    renderTranscript();
    if (settings.speak && "speechSynthesis" in window) {
      speechSynthesis.cancel();
      speechSynthesis.speak(new SpeechSynthesisUtterance(p.label));
    }
  }
  window.__hush?.onDecode?.(p);
}

function renderTranscript() {
  const t = $("transcript");
  t.textContent = transcriptWords.join(" ");
  t.scrollTop = t.scrollHeight;
}

function renderSegmenter() {
  const s = segmenter;
  const states = { uncalibrated: "Not calibrated", calibrating: `Calibrating… ${pct(s.progress)}`, idle: "Listening", active: "Word in progress" };
  $("segState").textContent = states[s.state];
  const max = settings.sensitivity * 2.5;
  const bar = $("segLevel");
  bar.style.width = `${Math.min(100, (s.level / max) * 100)}%`;
  bar.classList.toggle("active", s.state === "active");
  $("segThresh").style.left = `${(settings.sensitivity / max) * 100}%`;
}

// ------------------------------------------------------------------------------------------ evaluate

async function runEvaluation(ablate) {
  if (evaluating) return;
  const takes = trainingTakes();
  const labels = takes.map((t) => t.label), frames = takes.map((t) => t.frames);
  const fcfg = featureCfg(), ccfg = classifierCfg();
  if (!fcfg.groups.length) { toast("Pick at least one feature."); return; }
  evaluating = true;
  $("evalBtn").disabled = $("ablateBtn").disabled = true;
  const prog = $("evalProgress"), bar = prog.firstElementChild;
  prog.hidden = false;
  const status = (t, f) => { $("evalStatus").textContent = t; if (f != null) bar.style.width = `${f * 100}%`; };
  try {
    status("Measuring distances between takes…", 0);
    const report = await core.evaluate(frames.map((f) => core.extract(f, fcfg)), labels, ccfg, {
      permutations: settings.permutations, onProgress: (f) => status(`Measuring distances… ${pct(f)}`, f * (ablate ? 0.3 : 0.9)),
    });
    let rows = [];
    if (ablate) {
      const combos = core.allGroupCombinations();
      for (const [i, combo] of combos.entries()) {
        status(`Comparing features ${i + 1}/${combos.length}…`, 0.3 + (0.7 * i) / combos.length);
        const cfg = core.featureConfig({ groups: combo, deltas: fcfg.deltas, smoothing: fcfg.smoothing });
        const r = await core.evaluate(frames.map((f) => core.extract(f, cfg)), labels, ccfg, { permutations: Math.min(200, settings.permutations) });
        rows.push({ combo, accuracy: r.accuracy, p: r.pValue });
      }
      rows.sort((a, b) => b.accuracy - a.accuracy);
    }
    status(`Done · ${report.sampleCount} takes, ${report.labels.length} words`, 1);
    renderReport(report, rows);
    window.__hush?.onEvaluate?.(report, rows);
  } catch (e) {
    status(e instanceof core.EvaluationError ? e.message : `Failed: ${e.message}`, 0);
    $("evalResult").replaceChildren();
  } finally {
    evaluating = false;
    $("evalBtn").disabled = $("ablateBtn").disabled = false;
    setTimeout(() => { prog.hidden = true; }, 600);
  }
}

function renderReport(r, rows) {
  const stat = (label, value, cls = "") => el("div", { className: `stat ${cls}` }, el("small", { textContent: label }), el("strong", { textContent: value }));
  const card = el("div", { className: "card" },
    el("div", { className: "stats" },
      stat("Accuracy", pct(r.accuracy)), stat("Chance", pct(r.chance)),
      stat("Shuffled labels", pct(r.nullMean)), stat("p-value", r.pValue < 0.001 ? "<0.001" : r.pValue.toFixed(3))),
    el("div", { className: `verdict ${r.significant ? "good" : "meh"}`, textContent: r.verdict }),
    el("div", { className: "card-title", textContent: "Shuffled-label accuracies vs yours" }),
    histogram(r.nulls, r.accuracy));
  const table = el("table", { className: "confusion" });
  table.append(el("tr", {}, el("th"), ...r.labels.map((l) => el("th", { textContent: l })), el("th", { textContent: "recall" })));
  const peak = Math.max(1, ...r.confusion.flat());
  r.labels.forEach((l, i) => {
    table.append(el("tr", {}, el("th", { className: "rowh", textContent: l }), ...r.confusion[i].map((n, j) => {
      const td = el("td", { textContent: n || "" });
      if (n) td.style.background = `color-mix(in srgb, var(${i === j ? "--good" : "--bad"}) ${Math.round((n / peak) * 55) + 10}%, var(--surface-2))`;
      return td;
    }), el("td", { textContent: pct(r.recall(l)) })));
  });
  const conf = el("div", { className: "card" }, el("div", { className: "card-title", textContent: "Mouthed (rows) vs decoded (columns)" }), el("div", { className: "scroll-x" }, table));
  const out = [card, conf];
  if (rows.length) {
    const abl = el("div", { className: "abl" });
    for (const row of rows) {
      abl.append(el("span", { textContent: row.combo.map((g) => core.GROUP_TITLES[g]).join(" + ") }),
        el("span", { textContent: `${pct(row.accuracy)} · p ${row.p.toFixed(3)}` }),
        el("div", { className: "bar" }, Object.assign(el("div", { className: row.p < 0.05 ? "" : "ns" }), { style: `width:${row.accuracy * 100}%` })));
    }
    out.push(el("div", { className: "card" }, el("div", { className: "card-title", textContent: "Which lip features carry the signal" }), abl));
  }
  $("evalResult").replaceChildren(...out);
}

function histogram(nulls, observed) {
  const bins = 20, counts = new Array(bins).fill(0);
  for (const v of nulls) counts[Math.min(bins - 1, Math.floor(v * bins))]++;
  const peak = Math.max(1, ...counts), W = 300, H = 90, bw = W / bins;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "hist");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Shuffled-label accuracies cluster near ${pct(nulls.reduce((a, v) => a + v, 0) / Math.max(1, nulls.length))}; yours is ${pct(observed)}`);
  const add = (tag, attrs) => { const e = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); svg.append(e); return e; };
  counts.forEach((c, i) => { const h = (c / peak) * (H - 22); add("rect", { x: i * bw + 1, y: H - 14 - h, width: bw - 2, height: h, rx: 2, fill: "var(--muted)", opacity: 0.45 }); });
  const x = Math.min(W - 2, Math.max(2, observed * W));
  add("rect", { x: x - 1.5, y: 4, width: 3, height: H - 18, rx: 1.5, fill: "var(--accent)" });
  add("text", { x: 2, y: H - 2, "font-size": 10, fill: "var(--muted)" }).textContent = "0%";
  add("text", { x: W - 2, y: H - 2, "font-size": 10, fill: "var(--muted)", "text-anchor": "end" }).textContent = "100%";
  add("text", { x: Math.min(W - 20, Math.max(20, x)), y: H - 2, "font-size": 10, fill: "var(--accent)", "text-anchor": "middle", "font-weight": 700 }).textContent = "you";
  return svg;
}

// ------------------------------------------------------------------------------------------ rendering

function renderTrain() {
  if (selectedWord && !dataset.vocabulary.includes(selectedWord)) selectedWord = null;
  selectedWord ??= core.leastRecorded(dataset, settings.condition);
  const w = selectedWord;
  $("promptWord").textContent = !w ? "Add a word" : core.isNoiseLabel(w) ? "smile, yawn, swallow…" : w;
  $("wordChips").replaceChildren(...dataset.vocabulary.map((v) => {
    const b = el("button", { className: `word${core.isNoiseLabel(v) ? " noise" : ""}`, type: "button" }, v, el("b", { textContent: core.countTakes(dataset, v, settings.condition) }));
    b.setAttribute("role", "option");
    b.setAttribute("aria-selected", String(v === w));
    b.onclick = () => { selectedWord = v; renderTrain(); };
    return b;
  }));
  $("undoTake").disabled = !lastTake;
  renderData();
}

function renderData() {
  const byCond = Object.keys(core.CONDITIONS).map((c) => `${core.CONDITIONS[c]}: ${dataset.takes.filter((t) => t.condition === c).length}`);
  $("dataSummary").textContent = `${dataset.takes.length} takes across ${dataset.vocabulary.length} words. ${byCond.join(" · ")}`;
}

function showTab(tab) {
  const prev = settings.tab;
  settings.tab = tab;
  document.body.classList.toggle("feed-mode", tab === "feed");
  saveSettings();
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== `panel-${tab}`;
  $("holdBtn").hidden = !(tab === "train" || tab === "decode");
  document.documentElement.style.setProperty("--dock-h", tab === "train" || tab === "decode" ? "148px" : "80px");
  if (tab === "decode") { renderSegmenter(); $("autoBox").hidden = !settings.autoDetect; }
  if (tab === "feed") enterFeed(); else if (prev === "feed") leaveFeed();
}

let toastTimer = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
}

// ------------------------------------------------------------------------------------------ wiring

function bindControls() {
  const condSel = $("condition"), evalSel = $("evalCondition");
  for (const [k, v] of Object.entries(core.CONDITIONS)) condSel.append(el("option", { value: k, textContent: v }));
  evalSel.append(el("option", { value: "all", textContent: "All conditions" }));
  for (const [k, v] of Object.entries(core.CONDITIONS)) evalSel.append(el("option", { value: k, textContent: v }));
  condSel.value = settings.condition;
  condSel.onchange = () => { settings.condition = condSel.value; saveSettings(); selectedWord = null; renderTrain(); };
  evalSel.value = settings.evalCondition;
  evalSel.onchange = () => { settings.evalCondition = evalSel.value; saveSettings(); };

  // Hold-to-capture: pointer events cover touch, pen and mouse. Space works on a keyboard.
  const hold = $("holdBtn");
  hold.disabled = true;
  hold.addEventListener("pointerdown", (e) => { e.preventDefault(); hold.setPointerCapture?.(e.pointerId); beginCapture(); });
  for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) hold.addEventListener(ev, () => endCapture());
  hold.addEventListener("contextmenu", (e) => e.preventDefault());
  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat || e.target.closest("input, select, textarea")) return;
    e.preventDefault(); beginCapture();
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space" && capturing) { e.preventDefault(); endCapture(); } });
  document.addEventListener("visibilitychange", () => { if (document.hidden) endCapture(); });

  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => showTab(b.dataset.tab);
  $("startBtn").onclick = startCamera;

  $("addForm").onsubmit = (e) => {
    e.preventDefault();
    const w = $("newWord").value.trim();
    if (!w) return;
    if (!dataset.vocabulary.includes(w)) dataset.vocabulary.push(w);
    selectedWord = w;
    $("newWord").value = "";
    $("newWord").blur();
    saveDataset();
    renderTrain();
  };
  const autoAdvance = $("autoAdvance");
  autoAdvance.checked = settings.autoAdvance;
  autoAdvance.onchange = () => { settings.autoAdvance = autoAdvance.checked; saveSettings(); };
  $("undoTake").onclick = () => {
    if (!lastTake) return;
    dataset.takes = dataset.takes.filter((t) => t.id !== lastTake.id);
    selectedWord = lastTake.label;
    $("takeInfo").textContent = `Removed the last take of “${lastTake.label}”.`;
    lastTake = null;
    saveDataset();
    renderTrain();
  };
  $("removeWord").onclick = () => {
    const w = selectedWord;
    if (!w) return;
    const n = core.countTakes(dataset, w);
    if (!confirm(n ? `Remove “${w}” and its ${n} takes?` : `Remove “${w}”?`)) return;
    dataset.vocabulary = dataset.vocabulary.filter((v) => v !== w);
    dataset.takes = dataset.takes.filter((t) => t.label !== w);
    selectedWord = null;
    saveDataset();
    renderTrain();
  };

  // Check
  const toggles = $("groupToggles");
  for (const g of core.DEFAULT_GROUPS) {
    const b = el("button", { className: "toggle", type: "button", textContent: core.GROUP_TITLES[g] });
    b.setAttribute("aria-pressed", String(settings.groups.includes(g)));
    b.onclick = () => {
      settings.groups = settings.groups.includes(g) ? settings.groups.filter((x) => x !== g) : [...settings.groups, g];
      b.setAttribute("aria-pressed", String(settings.groups.includes(g)));
      saveSettings();
    };
    toggles.append(b);
  }
  const bindSelect = (id, key, parse = Number) => { const s = $(id); s.value = String(settings[key]); s.onchange = () => { settings[key] = parse(s.value); saveSettings(); }; };
  bindSelect("permutations", "permutations");
  bindSelect("kSel", "k");
  bindSelect("smoothSel", "smoothing");
  const deltas = $("deltas");
  deltas.checked = settings.deltas;
  deltas.onchange = () => { settings.deltas = deltas.checked; saveSettings(); };
  $("evalBtn").onclick = () => runEvaluation(false);
  $("ablateBtn").onclick = () => runEvaluation(true);

  // Decode
  const bindSwitch = (id, key, after) => { const s = $(id); s.checked = settings[key]; s.onchange = () => { settings[key] = s.checked; saveSettings(); after?.(); }; };
  bindSwitch("speak", "speak");
  bindSwitch("autoDetect", "autoDetect", () => {
    $("autoBox").hidden = !settings.autoDetect;
    if (settings.autoDetect && !segmenter.calibrated) segmenter.beginCalibration();
  });
  $("autoBox").hidden = !settings.autoDetect;
  $("calibrateBtn").onclick = () => segmenter.beginCalibration();
  const bindRange = (id, key, fmt) => {
    const r = $(id), out = $(`${id}Out`);
    r.value = settings[key];
    out.textContent = fmt(settings[key]);
    r.oninput = () => { settings[key] = Number(r.value); out.textContent = fmt(settings[key]); saveSettings(); };
  };
  bindRange("sens", "sensitivity", (v) => `${v.toFixed(1)}σ`);
  bindRange("maxd", "maxDistance", (v) => v.toFixed(2));
  bindRange("margin", "minMargin", (v) => pct(v));
  $("copyBtn").onclick = async () => {
    try { await navigator.clipboard.writeText(transcriptWords.join(" ")); toast("Copied"); } catch { toast("Copy not allowed here"); }
  };
  $("shareBtn").onclick = async () => {
    const text = transcriptWords.join(" ");
    if (navigator.share) { try { await navigator.share({ text }); } catch { /* cancelled */ } } else toast("Sharing isn't supported in this browser");
  };
  $("clearBtn").onclick = () => { transcriptWords.length = 0; renderTranscript(); };

  // Data
  $("exportBtn").onclick = exportDataset;
  $("importFile").onchange = importDataset;
  $("resetBtn").onclick = () => {
    if (!dataset.takes.length || !confirm(`Delete all ${dataset.takes.length} takes? Export first if you want a backup.`)) return;
    dataset.takes = [];
    lastTake = null;
    saveDataset();
    renderTrain();
  };
  const mirror = $("mirror");
  mirror.checked = settings.mirror;
  $("stage").classList.toggle("mirror", settings.mirror);
  mirror.onchange = () => { settings.mirror = mirror.checked; $("stage").classList.toggle("mirror", settings.mirror); saveSettings(); };
  $("flipBtn").onclick = async () => {
    settings.facing = settings.facing === "user" ? "environment" : "user";
    settings.mirror = settings.facing === "user";
    mirror.checked = settings.mirror;
    $("stage").classList.toggle("mirror", settings.mirror);
    saveSettings();
    if (running) { try { await openStream(); } catch (e) { toast(e.message); } }
  };
}

async function exportDataset() {
  const json = JSON.stringify(core.datasetToJSON(dataset));
  const name = `hush-dataset-${new Date().toISOString().slice(0, 10)}.json`;
  const file = new File([json], name, { type: "application/json" });
  if (navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: "Hush training data" }); return; } catch (e) { if (e.name === "AbortError") return; }
  }
  const a = el("a", { href: URL.createObjectURL(file), download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

async function importDataset(e) {
  const f = e.target.files?.[0];
  e.target.value = "";
  if (!f) return;
  try {
    const incoming = core.datasetFromJSON(JSON.parse(await f.text()));
    const known = new Set(dataset.takes.map((t) => t.id));
    const fresh = incoming.takes.filter((t) => !known.has(t.id));
    for (const w of incoming.vocabulary) if (!dataset.vocabulary.includes(w)) dataset.vocabulary.push(w);
    dataset.takes.push(...fresh);
    saveDataset();
    renderTrain();
    toast(`Imported ${fresh.length} takes`);
  } catch (err) {
    toast(`Import failed: ${err.message}`);
  }
}

async function init() {
  bindControls();
  try {
    const saved = await store.get("dataset");
    if (saved) dataset = core.datasetFromJSON(saved);
  } catch (e) { console.warn("No saved dataset", e); }
  renderTrain();
  showTab(["check", "data", "decode", "feed"].includes(settings.tab) ? settings.tab : "train");
  $("feedRefresh").onclick = () => { feed.loadedAt = 0; loadFeed(); };
  document.addEventListener("visibilitychange", () => { if (!document.hidden && settings.tab === "feed") loadFeed(); scheduleFeed(); });
  setInterval(() => { if (settings.tab === "feed") renderFeedStatus(); }, 30e3);
  if (settings.tab !== "feed") { loadFeed(); scheduleFeed(); }
  if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("sw.js").catch(() => {});
}


// ------------------------------------------------------------------------------------------ Truth Social feed
// feed.json is refreshed every ~5 min by .github/workflows/feed.yml on the `feed-data` branch.

const FEED_URL = (() => {
  const m = location.hostname.match(/^([^.]+)\.github\.io$/);
  const repo = location.pathname.split("/").filter(Boolean)[0];
  const [owner, name] = m && repo ? [m[1], repo] : ["mishalgovani-afk", "hush-web"];
  return `https://raw.githubusercontent.com/${owner}/${name}/feed-data/feed.json`;
})();
const feed = { posts: [], loadedAt: 0, timer: null, loading: false, error: null, meta: null };
const LAST_SEEN = "hush.feed.lastSeen";
const lastSeen = () => { try { return localStorage.getItem(LAST_SEEN) || ""; } catch { return ""; } };
const setLastSeen = (id) => { try { localStorage.setItem(LAST_SEEN, id); } catch { /* private mode */ } };
const newerId = (a, b) => (a.length !== b.length ? a.length > b.length : a > b);

function ago(iso) {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  if (s < 7 * 86400) return `${Math.floor(s / 86400)}d`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/** Plain text -> DOM with clickable https links. Never uses innerHTML with feed content. */
function linkify(text) {
  const frag = document.createDocumentFragment();
  let last = 0;
  for (const m of text.matchAll(/https?:\/\/[^\s<>"]+[^\s<>".,;:!?)\]'”’]/g)) {
    frag.append(text.slice(last, m.index));
    const url = m[0];
    if (/^https:\/\//i.test(url)) {
      const a = el("a", { href: url, target: "_blank", rel: "noopener noreferrer nofollow", textContent: url.replace(/^https:\/\/(www\.)?/, "").slice(0, 60) + (url.length > 68 ? "…" : "") });
      frag.append(a);
    } else frag.append(url);
    last = m.index + url.length;
  }
  frag.append(text.slice(last));
  return frag;
}

function renderPost(p, seen) {
  const fmt = (n) => (n == null ? null : n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}K` : String(n));
  const time = el("time", { dateTime: p.createdAt, textContent: ago(p.createdAt), title: new Date(p.createdAt).toLocaleString() });
  const card = el("article", { className: `post${seen && newerId(p.id, seen) ? " new" : ""}` },
    el("div", { className: "post-meta" }, el("span", { className: "avatar", textContent: "DT" }),
      el("span", {}, el("strong", { textContent: "Donald J. Trump" }), " @realDonaldTrump"), time));
  if (p.repost) card.append(el("p", { className: "post-repost", textContent: `↻ Reposted${p.repostOf ? ` from ${p.repostOf}` : ""}` }));
  const body = el("p", { className: `post-text${p.text ? "" : " empty"}` });
  if (p.text) body.append(linkify(p.text)); else body.textContent = p.media?.length ? "(media only)" : "(no text, open on Truth Social)";
  card.append(body);
  if (p.media?.length) {
    const grid = el("div", { className: "post-media" });
    for (const m of p.media) {
      const src = m.preview || (m.type === "image" ? m.url : null);
      const a = el("a", { href: m.url || p.url, target: "_blank", rel: "noopener noreferrer" });
      if (src) {
        const img = el("img", { src, alt: m.type === "video" ? "Video preview" : "Attached image", loading: "lazy", referrerPolicy: "no-referrer" });
        img.onerror = () => a.remove();
        a.append(img);
      }
      if (m.type === "video") a.append(el("span", { className: "play", textContent: "▶" }));
      grid.append(a);
    }
    card.append(grid);
  }
  const foot = el("div", { className: "post-foot" });
  for (const [icon, n] of [["💬", p.replies], ["↻", p.reposts], ["♥", p.likes]]) if (n != null) foot.append(el("span", { textContent: `${icon} ${fmt(n)}` }));
  foot.append(el("a", { href: p.url, target: "_blank", rel: "noopener noreferrer", textContent: "Open ↗" }));
  card.append(foot);
  return card;
}

function renderFeed() {
  const list = $("feedList");
  const seen = list.dataset.seen ?? lastSeen();
  list.dataset.seen = seen;
  if (!feed.posts.length) {
    list.replaceChildren(el("div", { className: "feed-empty", textContent: feed.error ? `Couldn't load the feed (${feed.error}).` : feed.loading ? "Loading posts…" : "No posts yet. The first update can take a few minutes." }));
  } else {
    list.replaceChildren(...feed.posts.map((p) => renderPost(p, seen)));
  }
  renderFeedStatus();
}

function renderFeedStatus() {
  const m = feed.meta;
  let t = feed.loading && !feed.posts.length ? "Loading…" : "";
  if (m?.updatedAt) t = `Updated ${ago(m.updatedAt)}${ago(m.updatedAt) === "just now" ? "" : " ago"} · ${feed.posts.length} posts${m.source ? ` · via ${m.source}` : ""}`;
  if (feed.error) t = `${t ? `${t} · ` : ""}offline (${feed.error})`;
  if (m?.checkedAt && m.updatedAt && new Date(m.checkedAt) - new Date(m.updatedAt) > 30 * 60e3) t += " · sources unreachable, showing last saved posts";
  $("feedStatus").textContent = t;
  for (const tEl of document.querySelectorAll("#feedList time")) tEl.textContent = ago(tEl.dateTime);
}

function updateFeedBadge() {
  const seen = lastSeen();
  const n = seen ? feed.posts.filter((p) => newerId(p.id, seen)).length : 0;
  const b = $("feedBadge");
  b.hidden = !n || settings.tab === "feed";
  b.textContent = n > 9 ? "9+" : String(n);
}

async function loadFeed() {
  if (feed.loading) return;
  feed.loading = true;
  if (settings.tab === "feed" && !feed.posts.length) renderFeed();
  try {
    const res = await fetch(`${FEED_URL}?t=${Math.floor(Date.now() / 30000)}`, { cache: "no-store" });
    if (res.status === 404) throw new Error("feed not published yet");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const before = feed.posts[0]?.id;
    feed.posts = Array.isArray(data.posts) ? data.posts.filter((p) => p && p.id && p.createdAt) : [];
    feed.meta = data;
    feed.error = null;
    feed.loadedAt = Date.now();
    if (settings.tab === "feed" && (before !== feed.posts[0]?.id || !$("feedList").children.length || $("feedList").querySelector(".feed-empty"))) renderFeed();
  } catch (e) {
    feed.error = e.message === "Failed to fetch" ? "no connection" : e.message;
    if (settings.tab === "feed") renderFeed();
  } finally {
    feed.loading = false;
    if (settings.tab === "feed") renderFeedStatus();
    updateFeedBadge();
  }
}

function scheduleFeed() {
  clearTimeout(feed.timer);
  const active = settings.tab === "feed" && !document.hidden;
  feed.timer = setTimeout(async () => { if (!document.hidden) await loadFeed(); scheduleFeed(); }, active ? 60e3 : 180e3);
}

function enterFeed() {
  const list = $("feedList");
  delete list.dataset.seen;          // highlight posts newer than the last visit
  renderFeed();
  if (Date.now() - feed.loadedAt > 20e3) loadFeed();
  if (feed.posts[0]) setLastSeen(feed.posts[0].id);
  updateFeedBadge();
  scheduleFeed();
}

function leaveFeed() {
  if (feed.posts[0]) setLastSeen(feed.posts[0].id);
  scheduleFeed();
}

// Test hooks: lets automated tests feed recorded lip frames without a real face.
window.__hush = {
  core,
  feed(frames) { for (const f of frames) handleFrame(f); },
  begin: beginCapture,
  end: endCapture,
  showTab,
  get dataset() { return dataset; },
  get transcript() { return transcriptWords.join(" "); },
  get segmenter() { return segmenter; },
  setRunning(v) { running = v; $("holdBtn").disabled = !v; if (v) $("startPanel").hidden = true; updateLive(); },
  selectWord(w) { selectedWord = w; renderTrain(); },
  evaluate: runEvaluation,
  feed,
  loadFeed,
  FEED_URL,
};

init();
