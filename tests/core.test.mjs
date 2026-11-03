// Run: node --test web/tests
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import * as core from "../core.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures/parity.json", import.meta.url)));
const frames = fx.takes.map((t) => t.frames.map(core.frameFromJSON));
const labels = fx.takes.map((t) => t.label);
const close = (a, b, tol = 1e-9, msg = "") => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${msg} ${a} != ${b}`);

for (const [name, c] of Object.entries(fx.configs)) {
  test(`parity with Python core: ${name}`, async () => {
    const cfg = core.featureConfig({ groups: c.groups, deltas: c.deltas, smoothing: c.smoothing });
    frames.forEach((f, i) => assert.equal(core.trim(f, cfg.trimThreshold).length, c.trim_lengths[i], `trim ${i}`));
    const seqs = frames.map((f) => core.extract(f, cfg));
    seqs.forEach((s, i) => {
      assert.deepEqual([s.n, s.d], c.seq_shapes[i]);
      close(s.data.reduce((a, v) => a + Math.abs(v), 0), c.seq_sums[i], 1e-9, `sum ${i}`);
    });
    const scales = core.fitScaler(seqs);
    scales.forEach((v, k) => close(v, c.scales[k], 1e-7, `scale ${k}`));
    const D = await core.distanceMatrix(seqs.map((s) => core.applyScaler(scales, s)), 0.2);
    for (let i = 0; i < D.length; i++) for (let j = 0; j < D.length; j++) close(D[i][j], c.D[i][j], 1e-7, `D[${i}][${j}]`);
    const r = await core.evaluateDistances(D, labels, { permutations: 0 });
    close(r.accuracy, c.loo_accuracy);
    assert.deepEqual(r.confusion, c.confusion);
    close((await core.evaluateDistances(D, labels, { k: 2, permutations: 0 })).accuracy, c.loo_accuracy_k2);
    const clf = new core.TemplateClassifier(labels.slice(4).map((l, i) => [l, seqs[i + 4]]), core.classifierConfig({ maxDistance: 99, minMargin: 0 }));
    seqs.slice(0, 4).forEach((s, i) => {
      const p = clf.classify(s), want = c.predictions[i];
      assert.deepEqual(p.ranked.map((x) => x[0]), want.ranked.map((x) => x[0]));
      p.ranked.forEach((x, k) => close(x[1], want.ranked[k][1], 1e-7));
      close(p.margin, want.margin, 1e-7);
    });
  });
}

test("parity: streaming segmenter finds the same words", () => {
  const seg = new core.Segmenter();
  seg.beginCalibration();
  const out = [];
  for (const f of fx.stream.map(core.frameFromJSON)) { const s = seg.process(f); if (s) out.push(s.map((x) => x.t)); }
  assert.deepEqual(out, fx.segments);
});

test("DTW properties", () => {
  const seq = (xs) => ({ n: xs.length, d: 1, data: Float64Array.from(xs) });
  const base = seq(Array.from({ length: 30 }, (_, i) => Math.sin((i / 29) * Math.PI)));
  const slow = seq(Array.from({ length: 40 }, (_, i) => Math.sin((i / 39) * Math.PI)));
  const other = seq(Array.from({ length: 30 }, (_, i) => -Math.sin((i / 29) * 2 * Math.PI)));
  close(core.dtw(base, base), 0);
  assert.ok(core.dtw(base, slow) < core.dtw(base, other) / 5);
  close(core.dtw(base, slow), core.dtw(slow, base));
  assert.equal(core.dtw(seq(new Array(20).fill(0)), seq(new Array(20).fill(10)), 0.2, 1), Infinity);
  close(core.dtw(seq(new Array(20).fill(0)), seq(new Array(20).fill(10))), 5);
  close(core.dtw(seq([1, 1, 1, 1, 1]), seq(new Array(50).fill(1)), 0.01), 0);
});

test("evaluation: significance and validation", async () => {
  const cfg = core.featureConfig();
  const seqs = frames.map((f) => core.extract(f, cfg));
  const r = await core.evaluate(seqs, labels, core.classifierConfig(), { permutations: 200 });
  assert.ok(r.accuracy > 0.8 && r.significant, r.verdict);
  const r2 = await core.evaluate(seqs, labels, core.classifierConfig(), { permutations: 200 });
  assert.deepEqual(r.nulls, r2.nulls);
  // Labels unrelated to the signal: shuffle them deterministically.
  const rand = core.rng(9), shuffled = [...labels];
  for (let i = shuffled.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; }
  const bad = await core.evaluate(seqs, shuffled, core.classifierConfig(), { permutations: 200 });
  assert.ok(bad.accuracy < r.accuracy);
  await assert.rejects(core.evaluate(seqs.slice(0, 2), ["a", "a"]), core.EvaluationError);
  await assert.rejects(core.evaluate(seqs.slice(0, 3), ["a", "a", "b"]), /Short: b/);
});

test("classifier rejects noise class, distance and empty", () => {
  const cfg = core.featureConfig();
  const seqs = frames.map((f) => core.extract(f, cfg));
  const ex = seqs.slice(4).map((s, i) => [labels[i + 4] === "w3" ? "_noise" : labels[i + 4], s]);
  const clf = new core.TemplateClassifier(ex, core.classifierConfig({ maxDistance: 99, minMargin: 0 }));
  const p = clf.classify(seqs[3]);
  assert.equal(p.label, null);
  assert.equal(p.rejection, "Matched a noise template");
  const strict = new core.TemplateClassifier(ex, core.classifierConfig({ maxDistance: 1e-9, minMargin: 0 }));
  assert.equal(strict.classify(seqs[0]).rejection, "Too far from every template");
  assert.equal(new core.TemplateClassifier([]).classify(seqs[0]).rejection, "No templates");
});

test("dataset JSON round trip and balancing", () => {
  const ds = core.newDataset();
  ds.vocabulary = ["yes", "no"];
  ds.takes.push({ id: core.newId(), label: "yes", condition: "mouthed", created: new Date().toISOString(), frames: frames[0] });
  assert.equal(core.leastRecorded(ds, "mouthed"), "no");
  const back = core.datasetFromJSON(JSON.parse(JSON.stringify(core.datasetToJSON(ds))));
  assert.deepEqual(back.vocabulary, ds.vocabulary);
  assert.equal(back.takes[0].frames.length, frames[0].length);
  assert.throws(() => core.datasetFromJSON({}), /Not a Hush dataset/);
});

test("landmark normalisation is rotation and scale invariant", () => {
  const pts = Array.from({ length: 478 }, (_, i) => [100 + 40 * Math.cos(i), 200 + 40 * Math.sin(i * 1.7)]);
  const blend = { jawOpen: 0.4 };
  const a = core.lipFrameFromLandmarks(pts, blend, 0);
  const th = 0.3, sc = 2.5;
  const moved = pts.map(([x, y]) => [sc * (Math.cos(th) * x - Math.sin(th) * y) + 7, sc * (Math.sin(th) * x + Math.cos(th) * y) - 3]);
  const b = core.lipFrameFromLandmarks(moved, blend, 0);
  a.shape.forEach((v, i) => close(b.shape[i], v, 1e-9));
  a.contour.forEach((v, i) => close(b.contour[i], v, 1e-9));
  assert.equal(a.blend[core.BLEND_NAMES.indexOf("jawOpen")], 0.4);
});
