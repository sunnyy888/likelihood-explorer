// Tests for the likelihood core. Run: node test.mjs   (loads <script id="core"> from index.html)
import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const LX = new Function("module", html.match(/<script id="core">([\s\S]*?)<\/script>/)[1] + "; return LX;")({});
let pass = 0, fail = 0;
const check = (n, ok, d = "") => { ok ? pass++ : fail++; console.log(`${ok ? "PASS" : "FAIL"}  ${n}${d ? ": " + d : ""}`); };

// 1. coin: the peak is at h/n and the curvature gives sqrt(p(1-p)/n)
{
  const h = 14, n = 20; let best = 0, bp = 0;
  for (let p = 0.001; p < 1; p += 0.0005) { const v = LX.coinLL(p, h, n); if (!bp || v > best) { best = v; bp = p; } }
  check("coin log-likelihood peaks at h/n = 0.7", Math.abs(bp - 0.7) < 1e-3, bp.toFixed(4));
  const e = 1e-4, curv = (LX.coinLL(0.7 + e, h, n) - 2 * LX.coinLL(0.7, h, n) + LX.coinLL(0.7 - e, h, n)) / (e * e);
  check("SE = 1/sqrt(-curvature) = sqrt(p(1-p)/n)", Math.abs(1 / Math.sqrt(-curv) - LX.coinSE(h, n)) < 1e-5 && Math.abs(LX.coinSE(h, n) - Math.sqrt(0.21 / 20)) < 1e-12, LX.coinSE(h, n).toFixed(5));
}
// 2. logistic: analytic gradient and Hessian vs finite differences
{
  const d = LX.logitData(LX.mulberry32(3), 80), b = [-2, 0.5], e = 1e-6, r = LX.logitLL(d, b);
  const gn = [0, 1].map(k => { const u = b.slice(), w = b.slice(); u[k] += e; w[k] -= e; return (LX.logitLL(d, u).ll - LX.logitLL(d, w).ll) / (2 * e); });
  const Hn = [0, 1].map(k => { const u = b.slice(), w = b.slice(); u[k] += e; w[k] -= e; return [0, 1].map(j => (LX.logitLL(d, u).g[j] - LX.logitLL(d, w).g[j]) / (2 * e)); });
  check("logistic gradient vs finite differences", Math.max(...gn.map((v, k) => Math.abs(v - r.g[k]))) < 1e-5);
  check("logistic Hessian vs finite differences", Math.max(...Hn.flatMap((row, k) => row.map((v, j) => Math.abs(v - r.H[k][j])))) < 1e-4);
}
// 3. Newton converges in a handful of steps; gradient ascent needs far more
{
  const d = LX.logitData(LX.mulberry32(3), 80), f = LX.fitLogit(d);
  let b = [0, 0], k = 0; while (Math.hypot(...LX.logitLL(d, b).g) > 1e-8 && k < 50) { b = LX.newtonStep(d, b); k++; }
  check("Newton reaches |gradient| < 1e-8 in under 10 steps", k < 10, `${k} steps`);
  let g = [0, 0], m = 0; while (Math.hypot(...LX.logitLL(d, g).g) > 1e-3 && m < 100000) { g = LX.gdStep(d, g, 10 ** -2.3); m++; }
  check("gradient ascent needs 50x+ more steps (tilted, stretched hill)", m > 50 * k, `${m} steps to |g| < 1e-3`);
  check("both land on the same maximum", Math.abs(g[0] - f.b[0]) < 0.05 && Math.abs(g[1] - f.b[1]) < 0.01);
}
// 4. curvature SEs vs the spread of refits: accurate in large samples, too small in small ones (a real effect)
function refit(n) {
  const d = LX.logitData(LX.mulberry32(3), n), f = LX.fitLogit(d), rng = LX.mulberry32(77), est = [];
  for (let s = 0; s < 1000; s++) { const y = d.x.map(x => (rng() < LX.sig(f.b[0] + f.b[1] * x) ? 1 : 0)); est.push(LX.fitLogit({ x: d.x, y }, f.b).b); }
  const sd = k => { const m = est.reduce((a, e) => a + e[k], 0) / est.length; return Math.sqrt(est.reduce((a, e) => a + (e[k] - m) ** 2, 0) / (est.length - 1)); };
  const Ci = LX.inv2(f.cov); let out = 0;
  est.forEach(e => { const a = e[0] - f.b[0], b = e[1] - f.b[1]; if (a * (Ci[0][0] * a + Ci[0][1] * b) + b * (Ci[1][0] * a + Ci[1][1] * b) > 5.991) out++; });
  return { f, sd: [sd(0), sd(1)], out: out / 1000, maxB1: Math.max(...est.map(e => Math.abs(e[1]))) };
}
{
  const big = refit(1000);
  check("1,000 students: curvature SE ≈ SD of 1,000 refits (within 5%)", [0, 1].every(k => Math.abs(big.sd[k] / big.f.se[k] - 1) < 0.05), `β1 SE ${big.f.se[1].toFixed(4)} vs SD ${big.sd[1].toFixed(4)}`);
  check("1,000 students: about 5% of refits outside the 95% ellipse", big.out > 0.035 && big.out < 0.075, `${(100 * big.out).toFixed(1)}%`);
  const small = refit(80);
  check("80 students: refits spread wider than curvature says (documented on the page), with no runaway fits", small.sd[1] / small.f.se[1] > 1.1 && small.maxB1 < 5, `ratio ${(small.sd[1] / small.f.se[1]).toFixed(2)}, ${(100 * small.out).toFixed(1)}% outside, max |β1| ${small.maxB1.toFixed(2)}`);
}
// 5. marching squares: contour of x^2 + y^2 = r^2 on a grid lies on the circle
{
  const N = 61, z = Array.from({ length: N }, (_, j) => Array.from({ length: N }, (_, i) => -(((i - 30) / 10) ** 2 + ((j - 30) / 10) ** 2)));
  const segs = LX.contour(z, -4), err = Math.max(...segs.flat().map(([i, j]) => Math.abs(Math.hypot((i - 30) / 10, (j - 30) / 10) - 2)));
  check("contour of a paraboloid is a circle of the right radius", segs.length > 40 && err < 0.01, `${segs.length} segments, max radius error ${err.toFixed(4)}`);
}
// 6. conjoint: fit recovers the synthetic truth; matches HW4 2B on the real file if present locally
{
  const d = LX.cjSynthetic(2026), f = LX.cjFit(d), truth = [4.4, 0.8, 0.47, -0.0285];
  const wald = f.b.map((b, k) => ((b - truth[k]) / f.se[k]) ** 2).reduce((a, b) => a + b);
  check("synthetic conjoint: converges and lands within sampling error of the truth", f.maxGrad < 1e-6 && wald < 16, `max |g| ${f.maxGrad.toExponential(1)}, sum z² = ${wald.toFixed(2)}`);
  const file = `${homedir()}/larkspur_hw4/HW4-larkspur_conjoint.csv`;
  if (existsSync(file)) {
    const lines = readFileSync(file, "utf8").trim().split(/\r?\n/), head = lines.shift().split(",");
    const rows = lines.map(l => { const c = l.split(","), o = {}; head.forEach((h, i) => (o[h] = h === "resp_id" ? c[i] : +c[i])); return o; });
    const hw = LX.cjFit(LX.cjFromRows(rows)), want = [4.3804, 0.8019, 0.4738, -0.028514], se = [0.3070, 0.0818, 0.0802, 0.001925];
    check("HW4 conjoint file: MLE and SEs match my HW4 2B answers", hw.b.every((b, k) => Math.abs(b - want[k]) < (k === 3 ? 5e-7 : 5e-5)) && hw.se.every((s, k) => Math.abs(s - se[k]) < (k === 3 ? 5e-7 : 5e-5)) && Math.abs(hw.ll + 1420.58) < 0.01,
      hw.b.map((b, k) => b.toFixed(k === 3 ? 6 : 4)).join(", ") + `, LL ${hw.ll.toFixed(2)}`);
  } else console.log("SKIP  HW4 conjoint file not found locally");
}
console.log(`\n${pass}/${pass + fail} PASS`); process.exit(fail ? 1 : 0);
