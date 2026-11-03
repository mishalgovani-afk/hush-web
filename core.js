// Hush Web decoder core: lip features, DTW template matching, honest evaluation, streaming segmentation.
// A line-for-line port of windows/hushcam/core.py (itself a port of the macOS HushCore).
// No DOM access: runs in the browser and under `node --test`.

export const BLEND_NAMES = [
  "jawForward", "jawLeft", "jawRight", "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
  "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
  "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight", "mouthRollLower",
  "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight",
  "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight", "cheekPuff",
];
export const SHAPE_NAMES = ["width", "inner_height", "outer_height", "aspect", "jaw_drop"];
// Face-mesh indices: 12 outer + 8 inner lip points.
export const CONTOUR_POINTS = [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, 78, 82, 13, 312, 308, 317, 14, 87];
export const LIP_OUTLINE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
export const LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

export const GROUPS = { shape: SHAPE_NAMES.length, contour: CONTOUR_POINTS.length * 2, blend: BLEND_NAMES.length };
export const GROUP_TITLES = { shape: "Lip geometry", contour: "Lip contour", blend: "Mouth blendshapes" };
export const DEFAULT_GROUPS = ["shape", "contour", "blend"];
export const CONDITIONS = { normal: "Normal speech", whisper: "Whisper", mouthed: "Silent mouthing" };
export const STARTER_VOCABULARY = ["yes", "no", "hello", "stop", "next", "_noise"];

// ---------------------------------------------------------------------------------------------
// Frames

/** @typedef {{t:number, shape:number[], contour:number[], blend:number[]}} LipFrame */

const L_EYE = 33, R_EYE = 263, MOUTH_L = 61, MOUTH_R = 291, UP_IN = 13, LO_IN = 14, UP_OUT = 0, LO_OUT = 17, NOSE = 1, CHIN = 152;

/** points: array of [x, y] in pixels. blend: {name: score}. */
export function lipFrameFromLandmarks(points, blend, t) {
  const [lex, ley] = points[L_EYE], [rex, rey] = points[R_EYE];
  const iod = Math.hypot(rex - lex, rey - ley) || 1;
  const roll = Math.atan2(rey - ley, rex - lex);
  const c = Math.cos(-roll), s = Math.sin(-roll);
  const cx = (points[MOUTH_L][0] + points[MOUTH_R][0]) / 2, cy = (points[MOUTH_L][1] + points[MOUTH_R][1]) / 2;
  const dist = (i, j) => Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) / iod;
  const width = dist(MOUTH_L, MOUTH_R), inner = dist(UP_IN, LO_IN), outer = dist(UP_OUT, LO_OUT);
  const shape = [width, inner, outer, width > 0 ? inner / width : 0, dist(NOSE, CHIN)];
  const contour = [];
  for (const i of CONTOUR_POINTS) {
    const dx = points[i][0] - cx, dy = points[i][1] - cy;
    contour.push((c * dx - s * dy) / iod, (s * dx + c * dy) / iod);
  }
  return { t, shape, contour, blend: BLEND_NAMES.map((n) => +(blend[n] ?? 0)) };
}

const r5 = (x) => Math.round(x * 1e5) / 1e5;
export const frameToJSON = (f) => ({ t: Math.round(f.t * 1e4) / 1e4, s: f.shape.map(r5), c: f.contour.map(r5), b: f.blend.map(r5) });
export const frameFromJSON = (d) => ({ t: +d.t, shape: d.s.map(Number), contour: d.c.map(Number), blend: d.b.map(Number) });
export const duration = (frames) => (frames.length > 1 ? frames[frames.length - 1].t - frames[0].t : 0);

// ---------------------------------------------------------------------------------------------
// Features. A sequence is {n, d, data: Float64Array(n*d)} row-major.

export function featureConfig(o = {}) {
  return { groups: o.groups ?? DEFAULT_GROUPS, trimThreshold: o.trimThreshold ?? 0.12, smoothing: o.smoothing ?? 3, deltas: o.deltas ?? false };
}
export const dimension = (cfg) => cfg.groups.reduce((a, g) => a + GROUPS[g], 0) * (cfg.deltas ? 2 : 1);

export function allGroupCombinations() {
  const names = Object.keys(GROUPS), out = [];
  for (let m = 1; m < 1 << names.length; m++) out.push(names.filter((_, i) => m & (1 << i)));
  return out.sort((a, b) => a.length - b.length || a.join().localeCompare(b.join()));
}

function rawRows(frames, groups) {
  const key = { shape: "shape", contour: "contour", blend: "blend" };
  return frames.map((f) => groups.flatMap((g) => f[key[g]]));
}

function smoothRows(rows, window) {
  if (window <= 1 || rows.length < 2) return rows;
  const half = Math.floor(window / 2), d = rows[0].length;
  return rows.map((_, i) => {
    const lo = Math.max(0, i - half), hi = Math.min(rows.length - 1, i + half), acc = new Array(d).fill(0);
    for (let j = lo; j <= hi; j++) for (let k = 0; k < d; k++) acc[k] += rows[j][k];
    return acc.map((v) => v / (hi - lo + 1));
  });
}

/** Mouth deviation from the resting pose (median of the first/last 3 frames), scaled to [0, 1]. */
export function activity(frames) {
  const x = rawRows(frames, ["shape", "blend"]);
  if (!x.length) return [];
  const edge = [...x.slice(0, 3), ...x.slice(-3)];
  const ref = x[0].map((_, k) => median(edge.map((r) => r[k])));
  const dev = smoothRows(x.map((r) => [Math.hypot(...r.map((v, k) => v - ref[k]))]), 3).map((r) => r[0]);
  const lo = Math.min(...dev), hi = Math.max(...dev);
  return hi - lo > 0 ? dev.map((v) => (v - lo) / (hi - lo)) : dev.map(() => 0);
}

function median(a) {
  const s = [...a].sort((p, q) => p - q), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function trim(frames, threshold, padding = 2, minimum = 6) {
  if (threshold <= 0 || frames.length <= minimum) return frames;
  const e = activity(frames);
  const first = e.findIndex((v) => v >= threshold);
  if (first < 0) return frames;
  let last = e.length - 1;
  while (e[last] < threshold) last--;
  const lo = Math.max(0, first - padding), hi = Math.min(frames.length - 1, last + padding);
  return hi - lo + 1 < minimum ? frames : frames.slice(lo, hi + 1);
}

/** Frames -> mean-centred feature sequence. */
export function extract(frames, cfg = featureConfig()) {
  const d = dimension(cfg);
  if (!cfg.groups.length) return { n: 0, d: 0, data: new Float64Array(0) };
  let rows = rawRows(trim(frames, cfg.trimThreshold), cfg.groups);
  if (!rows.length) return { n: 0, d, data: new Float64Array(0) };
  rows = smoothRows(rows, cfg.smoothing);
  if (cfg.deltas) rows = rows.map((r, i) => r.concat(i ? r.map((v, k) => v - rows[i - 1][k]) : r.map(() => 0)));
  const n = rows.length, data = new Float64Array(n * d);
  for (let k = 0; k < d; k++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += rows[i][k];
    mean /= n;
    for (let i = 0; i < n; i++) data[i * d + k] = rows[i][k] - mean;
  }
  return { n, d, data };
}

/** Per-channel 1/sigma over training frames (label-free, so no leakage into evaluation). */
export function fitScaler(seqs) {
  const use = seqs.filter((s) => s.n > 0);
  if (!use.length) return null;
  const d = use[0].d, sum = new Float64Array(d), sq = new Float64Array(d);
  let n = 0;
  for (const s of use) for (let i = 0; i < s.n; i++) for (let k = 0; k < d; k++) { const v = s.data[i * d + k]; sum[k] += v; sq[k] += v * v; }
  for (const s of use) n += s.n;
  return Array.from({ length: d }, (_, k) => { const m = sum[k] / n, sd = Math.sqrt(Math.max(0, sq[k] / n - m * m)); return sd > 1e-9 ? 1 / sd : 1; });
}

export function applyScaler(scales, s) {
  if (!scales || scales.length !== s.d) return s;
  const data = new Float64Array(s.data.length);
  for (let i = 0; i < s.n; i++) for (let k = 0; k < s.d; k++) data[i * s.d + k] = s.data[i * s.d + k] * scales[k];
  return { n: s.n, d: s.d, data };
}

// ---------------------------------------------------------------------------------------------
// DTW: normalised by path length and by dimension (per-frame cost is the RMS channel difference).

export function dtw(a, b, band = 0.2, cutoff = Infinity) {
  const n = a.n, m = b.n;
  if (!n || !m || a.d !== b.d) return Infinity;
  const d = a.d, A = a.data, B = b.data, norm = n + m;
  const w = Math.max(Math.ceil(band * Math.max(n, m)), Math.abs(n - m), 1);
  let prev = new Float64Array(m + 1).fill(Infinity), cur = new Float64Array(m + 1);
  prev[0] = 0;
  for (let i = 1; i <= n; i++) {
    cur.fill(Infinity);
    const lo = Math.max(1, i - w), hi = Math.min(m, i + w), ai = (i - 1) * d;
    let rowMin = Infinity;
    for (let j = lo; j <= hi; j++) {
      const bj = (j - 1) * d;
      let s = 0;
      for (let k = 0; k < d; k++) { const x = A[ai + k] - B[bj + k]; s += x * x; }
      const v = Math.sqrt(s / d) + Math.min(prev[j], prev[j - 1], cur[j - 1]);
      cur[j] = v;
      if (v < rowMin) rowMin = v;
    }
    if (rowMin / norm > cutoff) return Infinity;
    [prev, cur] = [cur, prev];
  }
  return prev[m] / norm;
}

/** Pairwise distances. Async + chunked so a phone's UI stays responsive; onProgress(fraction). */
export async function distanceMatrix(seqs, band = 0.2, onProgress) {
  const n = seqs.length, D = Array.from({ length: n }, () => new Float64Array(n));
  const total = (n * (n - 1)) / 2;
  let done = 0, last = Date.now();
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) { D[i][j] = D[j][i] = dtw(seqs[i], seqs[j], band); done++; }
    if (Date.now() - last > 30) { last = Date.now(); onProgress?.(done / total); await new Promise((r) => setTimeout(r, 0)); }
  }
  onProgress?.(1);
  return D;
}

// ---------------------------------------------------------------------------------------------
// Classifier

export function classifierConfig(o = {}) {
  return { band: o.band ?? 0.2, k: o.k ?? 1, maxDistance: o.maxDistance ?? 1.0, minMargin: o.minMargin ?? 0.08 };
}
export const isNoiseLabel = (l) => l.startsWith("_");

export class TemplateClassifier {
  constructor(examples, cfg = classifierConfig()) {
    const use = examples.filter(([, s]) => s.n > 0);
    this.scales = fitScaler(use.map(([, s]) => s));
    this.templates = use.map(([l, s]) => [l, applyScaler(this.scales, s)]);
    this.cfg = cfg;
  }

  classify(raw) {
    if (!this.templates.length || !raw.n) return { label: null, ranked: [], margin: 0, rejection: "No templates" };
    const seq = applyScaler(this.scales, raw), per = new Map();
    for (const [l, t] of this.templates) {
      if (!per.has(l)) per.set(l, []);
      per.get(l).push(dtw(seq, t, this.cfg.band));
    }
    const k = Math.max(1, this.cfg.k);
    const ranked = [...per].map(([l, ds]) => { const s = ds.sort((p, q) => p - q).slice(0, k); return [l, s.reduce((a, v) => a + v, 0) / s.length]; })
      .sort((p, q) => p[1] - q[1] || p[0].localeCompare(q[0]));
    const best = ranked[0];
    const margin = ranked.length > 1 && isFinite(ranked[1][1]) && ranked[1][1] > 0 ? (ranked[1][1] - best[1]) / ranked[1][1] : 1;
    let rejection = null;
    if (isNoiseLabel(best[0])) rejection = "Matched a noise template";
    else if (!(best[1] <= this.cfg.maxDistance)) rejection = "Too far from every template";
    else if (margin < this.cfg.minMargin) rejection = "Too close to call";
    return { label: rejection ? null : best[0], ranked, margin, rejection };
  }
}

// ---------------------------------------------------------------------------------------------
// Evaluation: leave-one-out accuracy + label-permutation test.

export class EvaluationError extends Error {}

export function looPredict(D, y, classes, k = 1) {
  const n = y.length, out = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    let best = Infinity, bestC = -1;
    for (let c = 0; c < classes; c++) {
      const ds = [];
      for (let j = 0; j < n; j++) if (j !== i && y[j] === c) ds.push(D[i][j]);
      if (!ds.length) continue;
      let score;
      if (k === 1) score = Math.min(...ds);
      else { const s = ds.sort((p, q) => p - q).slice(0, k); score = s.reduce((a, v) => a + v, 0) / s.length; }
      if (score < best) { best = score; bestC = c; }
    }
    out[i] = bestC;
  }
  return out;
}

/** Deterministic PRNG (mulberry32) so permutation tests are reproducible. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export async function evaluateDistances(D, labels, { k = 1, permutations = 500, seed = 0x5eed } = {}) {
  const names = [...new Set(labels)].sort(), index = new Map(names.map((n, i) => [n, i]));
  const y = Int32Array.from(labels, (l) => index.get(l)), C = names.length;
  const acc = (p, t) => p.reduce((a, v, i) => a + (v === t[i] ? 1 : 0), 0) / t.length;
  const pred = looPredict(D, y, C, k), accuracy = acc(pred, y);
  const confusion = names.map(() => new Array(C).fill(0));
  y.forEach((t, i) => { if (pred[i] >= 0) confusion[t][pred[i]]++; });
  const rand = rng(seed), ys = Int32Array.from(y), nulls = new Float64Array(permutations);
  let atLeast = 0, last = Date.now();
  for (let r = 0; r < permutations; r++) {
    for (let i = ys.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [ys[i], ys[j]] = [ys[j], ys[i]]; }
    nulls[r] = acc(looPredict(D, ys, C, k), ys);
    if (nulls[r] >= accuracy - 1e-12) atLeast++;
    if (Date.now() - last > 30) { last = Date.now(); await new Promise((res) => setTimeout(res, 0)); }
  }
  const counts = names.map((_, c) => y.filter((v) => v === c).length);
  return makeReport({ labels: names, sampleCount: y.length, accuracy, chance: 1 / C, majority: Math.max(...counts) / y.length,
    pValue: (atLeast + 1) / (permutations + 1), permutations, nulls: Array.from(nulls), confusion });
}

function makeReport(r) {
  r.nullMean = r.nulls.length ? r.nulls.reduce((a, v) => a + v, 0) / r.nulls.length : r.chance;
  r.significant = r.pValue < 0.05;
  r.recall = (l) => { const i = r.labels.indexOf(l), row = r.confusion[i], s = row.reduce((a, v) => a + v, 0); return s ? row[i] / s : 0; };
  const pct = (x) => `${Math.round(x * 100)}%`;
  r.verdict = r.significant
    ? `Your lip movements separate these words: ${pct(r.accuracy)} vs ${pct(r.nullMean)} expected by chance (p = ${r.pValue.toFixed(3)}).`
    : `No evidence yet: ${pct(r.accuracy)} is within what shuffled labels reach (${pct(r.nullMean)}, p = ${r.pValue.toFixed(3)}). Record more takes or try other features.`;
  return r;
}

export async function evaluate(seqs, labels, cfg = classifierConfig(), { permutations = 500, seed = 0x5eed, onProgress } = {}) {
  const names = [...new Set(labels)].sort();
  if (names.length < 2) throw new EvaluationError("Record at least two different words.");
  const short = names.filter((n) => labels.filter((l) => l === n).length < 2);
  if (short.length) throw new EvaluationError(`Each word needs at least two takes. Short: ${short.join(", ")}.`);
  const scales = fitScaler(seqs), scaled = seqs.map((s) => applyScaler(scales, s));
  const D = await distanceMatrix(scaled, cfg.band, onProgress);
  return evaluateDistances(D, labels, { k: cfg.k, permutations, seed });
}

// ---------------------------------------------------------------------------------------------
// Streaming segmentation: calibrate on a resting mouth, emit spans where the mouth departs from rest.

export function segmenterConfig(o = {}) {
  return { sensitivity: 4, releaseRatio: 0.5, minOnFrames: 2, prerollFrames: 5, hangoverFrames: 8, minFrames: 8, maxFrames: 150, smoothing: 0.5, calibrationFrames: 60, ...o };
}

export class Segmenter {
  constructor(cfg = segmenterConfig()) { this.cfg = cfg; this.state = "uncalibrated"; this.level = 0; this.progress = 0; this.reset(); }
  get calibrated() { return this.state === "idle" || this.state === "active"; }
  static vec(f) { return f.shape.concat(f.blend); }
  beginCalibration() { this.cal = []; this.state = "calibrating"; this.progress = 0; this.reset(); }
  reset() { this.pre = []; this.buf = []; this.above = this.below = 0; this.level = 0; if (this.state === "active") this.state = "idle"; }
  setRest(mu, sd) { this.mu = mu; this.sd = sd.map((v, i) => Math.max(v, 0.02 * Math.abs(mu[i]), 1e-3)); this.state = "idle"; this.reset(); }

  process(f) {
    const c = this.cfg;
    if (this.state === "uncalibrated") return null;
    if (this.state === "calibrating") {
      this.cal.push(Segmenter.vec(f));
      this.progress = this.cal.length / c.calibrationFrames;
      if (this.cal.length >= c.calibrationFrames) {
        const d = this.cal[0].length, n = this.cal.length;
        const mu = Array.from({ length: d }, (_, k) => this.cal.reduce((a, r) => a + r[k], 0) / n);
        const sd = mu.map((m, k) => Math.sqrt(this.cal.reduce((a, r) => a + (r[k] - m) ** 2, 0) / n));
        this.setRest(mu, sd);
      }
      return null;
    }
    const v = Segmenter.vec(f);
    const z = Math.sqrt(v.reduce((a, x, k) => a + ((x - this.mu[k]) / this.sd[k]) ** 2, 0) / v.length);
    this.level += c.smoothing * (z - this.level);
    if (this.state === "idle") {
      this.pre.push(f);
      const keep = c.prerollFrames + c.minOnFrames;
      if (this.pre.length > keep) this.pre = this.pre.slice(-keep);
      this.above = this.level > c.sensitivity ? this.above + 1 : 0;
      if (this.above >= c.minOnFrames) { this.state = "active"; this.buf = this.pre; this.pre = []; this.below = 0; }
      return null;
    }
    this.buf.push(f);
    this.below = this.level < c.sensitivity * c.releaseRatio ? this.below + 1 : 0;
    if (this.below < c.hangoverFrames && this.buf.length < c.maxFrames) return null;
    const seg = this.buf.slice(0, this.buf.length - Math.max(0, this.below - 2));
    this.buf = []; this.above = this.below = 0; this.state = "idle";
    return seg.length >= c.minFrames ? seg : null;
  }
}

// ---------------------------------------------------------------------------------------------
// Dataset (same JSON as the Windows HushCam app, so training data moves between them).

export function newDataset() { return { vocabulary: [...STARTER_VOCABULARY], takes: [] }; }
export const countTakes = (ds, label, cond = null) => ds.takes.filter((t) => t.label === label && (!cond || t.condition === cond)).length;
export const filteredTakes = (ds, cond = null) => ds.takes.filter((t) => ds.vocabulary.includes(t.label) && (!cond || t.condition === cond));

export function leastRecorded(ds, cond = null, exclude = null) {
  const pool = ds.vocabulary.filter((w) => w !== exclude);
  const use = pool.length ? pool : ds.vocabulary;
  let best = null, bestN = Infinity;
  for (const w of use) { const n = countTakes(ds, w, cond); if (n < bestN) { best = w; bestN = n; } }
  return best;
}

export function newId() { return Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join(""); }

export function datasetToJSON(ds) {
  return { version: 1, kind: "hushcam-lips", vocabulary: ds.vocabulary,
    takes: ds.takes.map((t) => ({ id: t.id, label: t.label, condition: t.condition, created: t.created, frames: t.frames.map(frameToJSON) })) };
}

export function datasetFromJSON(d) {
  if (!d || !Array.isArray(d.takes)) throw new Error("Not a Hush dataset file.");
  return { vocabulary: Array.isArray(d.vocabulary) ? d.vocabulary.map(String) : [...STARTER_VOCABULARY],
    takes: d.takes.map((t) => ({ id: String(t.id), label: String(t.label), condition: String(t.condition), created: String(t.created), frames: t.frames.map(frameFromJSON) })) };
}
