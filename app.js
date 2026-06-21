// ═══════════════════════════════════════════════════════
//  app.js — ML Regression Visualizer (3 models)
// ═══════════════════════════════════════════════════════

// ── Chart.js default font ────────────────────────────
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#8080a8';

// ── Global chart instances ───────────────────────────
let lrChart = null, prChart = null, lwChart = null;

// ── Active tab tracking ──────────────────────────────
let activeTab = 'lr';

// ── LWLR state ───────────────────────────────────────
let lwState = { tau: 0.5, sigma: 0.2, showTrue: false };
let lwData = generateLwlrData(0.2);

// ════════════════════════════════════════════════════
//  1. LINEAR REGRESSION
// ════════════════════════════════════════════════════
function linReg(x, y) {
  const n = x.length;
  const xm = x.reduce((a, b) => a + b, 0) / n;
  const ym = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (x[i] - xm) * (y[i] - ym); den += (x[i] - xm) ** 2; }
  const slope = num / den;
  const intercept = ym - slope * xm;
  return { slope, intercept };
}

function r2Score(y, yp) {
  const ym = y.reduce((a, b) => a + b, 0) / y.length;
  const ss_res = y.reduce((s, yi, i) => s + (yi - yp[i]) ** 2, 0);
  const ss_tot = y.reduce((s, yi) => s + (yi - ym) ** 2, 0);
  return 1 - ss_res / ss_tot;
}

function rmse(y, yp) {
  return Math.sqrt(y.reduce((s, yi, i) => s + (yi - yp[i]) ** 2, 0) / y.length);
}

function initLRChart() {
  const { x, y } = californiaData;
  const { slope, intercept } = linReg(x, y);

  const xMin = Math.min(...x), xMax = Math.max(...x);
  const lineX = [xMin, xMax];
  const lineY = lineX.map(xi => slope * xi + intercept);
  const predY = x.map(xi => slope * xi + intercept);

  // Stats
  const r2 = r2Score(y, predY);
  const rm = rmse(y, predY);
  document.getElementById('lr-r2').textContent = r2.toFixed(4);
  document.getElementById('lr-rmse').textContent = rm.toFixed(4);
  document.getElementById('lr-pts').textContent = x.length;

  const ctx = document.getElementById('lrChart').getContext('2d');
  lrChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          type: 'scatter',
          label: 'Test Data',
          data: x.map((xi, i) => ({ x: xi, y: y[i] })),
          backgroundColor: 'rgba(56,189,248,0.45)',
          borderColor: 'rgba(56,189,248,0.2)',
          borderWidth: 1,
          pointRadius: 5,
          pointHoverRadius: 7
        },
        {
          type: 'line',
          label: 'Linear Fit',
          data: lineX.map((xi, i) => ({ x: xi, y: lineY[i] })),
          borderColor: '#38bdf8',
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0,
          fill: false
        }
      ]
    },
    options: chartOptions('Median Income (MedInc)', 'House Value ($100k)', '#38bdf8')
  });
}

// ════════════════════════════════════════════════════
//  2. POLYNOMIAL REGRESSION
// ════════════════════════════════════════════════════
function polyFeatures(x, degree) {
  return x.map(xi => Array.from({ length: degree + 1 }, (_, d) => Math.pow(xi, d)));
}

function matMul(A, B) {
  const r = A.length, c = B[0].length, k = B.length;
  return Array.from({ length: r }, (_, i) =>
    Array.from({ length: c }, (_, j) =>
      A[i].reduce((s, _, l) => s + A[i][l] * B[l][j], 0)));
}

function matTranspose(A) {
  return A[0].map((_, j) => A.map(row => row[j]));
}

function matInv2(M) {
  // Using numeric pseudo-inverse via Gaussian elimination
  const n = M.length;
  const aug = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => i === j ? 1 : 0)]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[maxRow][col])) maxRow = r;
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const piv = aug[col][col];
    if (Math.abs(piv) < 1e-12) continue;
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= piv;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = aug[r][col];
      for (let j = 0; j < 2 * n; j++) aug[r][j] -= f * aug[col][j];
    }
  }
  return aug.map(row => row.slice(n));
}

function polyFit(x, y, degree) {
  const X = polyFeatures(x, degree);
  const Xt = matTranspose(X);
  const XtX = matMul(Xt, X);
  const XtXinv = matInv2(XtX);
  const Xty = Xt.map(row => row.reduce((s, v, i) => s + v * y[i], 0));
  const theta = XtXinv.map(row => row.reduce((s, v, i) => s + v * Xty[i], 0));
  return theta;
}

function polyPredict(theta, x) {
  return x.map(xi => theta.reduce((s, t, d) => s + t * Math.pow(xi, d), 0));
}

let polyDegree = 2;

function buildPRChart(degree) {
  const { hp, mpg } = mpgData;
  const theta = polyFit(hp, mpg, degree);
  const predMpg = polyPredict(theta, hp);
  const r2 = r2Score(mpg, predMpg);
  const rm = rmse(mpg, predMpg);

  // Dense curve
  const xMin = Math.min(...hp) - 2, xMax = Math.max(...hp) + 2;
  const curveX = Array.from({ length: 200 }, (_, i) => xMin + (xMax - xMin) * i / 199);
  const curveY = polyPredict(theta, curveX);

  document.getElementById('pr-r2').textContent = r2.toFixed(4);
  document.getElementById('pr-rmse').textContent = rm.toFixed(4);
  document.getElementById('pr-deg').textContent = degree;
  document.getElementById('degreeLabel').textContent = degree;

  const datasets = [
    {
      type: 'scatter',
      label: 'MPG Data',
      data: hp.map((xi, i) => ({ x: xi, y: mpg[i] })),
      backgroundColor: 'rgba(129,140,248,0.45)',
      borderColor: 'rgba(129,140,248,0.2)',
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7
    },
    {
      type: 'line',
      label: `Degree ${degree} Fit`,
      data: curveX.map((xi, i) => ({ x: xi, y: curveY[i] })),
      borderColor: '#818cf8',
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.4,
      fill: false
    }
  ];

  if (!prChart) {
    const ctx = document.getElementById('prChart').getContext('2d');
    prChart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: chartOptions('Horsepower', 'Miles Per Gallon (MPG)', '#818cf8')
    });
  } else {
    prChart.data.datasets = datasets;
    prChart.update('active');
  }
}

// ════════════════════════════════════════════════════
//  3. LWLR
// ════════════════════════════════════════════════════
function lwlrKernel(x0, X, tau) {
  return X.map(xi => Math.exp(-(xi - x0) ** 2 / (2 * tau * tau)));
}

function lwlrPredict(x0, X, y, tau) {
  const w = lwlrKernel(x0, X, tau);
  const n = X.length;
  let a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
  for (let i = 0; i < n; i++) {
    a00 += w[i]; a01 += w[i] * X[i]; a11 += w[i] * X[i] * X[i];
    b0 += w[i] * y[i]; b1 += w[i] * X[i] * y[i];
  }
  const det = a00 * a11 - a01 * a01;
  if (Math.abs(det) < 1e-12) return 0;
  const t0 = (a11 * b0 - a01 * b1) / det;
  const t1 = (-a01 * b0 + a00 * b1) / det;
  return t0 + t1 * x0;
}

function lwlrCurve(X, y, tau, n = 300) {
  const Xt = Array.from({ length: n }, (_, i) => -3 + 6 * i / (n - 1));
  const yp = Xt.map(x => lwlrPredict(x, X, y, tau));
  return { Xt, yp };
}

function lwlrRMSE(X, y, tau) {
  const pred = X.map(xi => lwlrPredict(xi, X, y, tau));
  const trueFn = X.map(xi => Math.sin(xi));
  return Math.sqrt(pred.reduce((s, p, i) => s + (p - trueFn[i]) ** 2, 0) / X.length).toFixed(4);
}

function buildLWChart() {
  const { X, y } = lwData;
  const { Xt, yp } = lwlrCurve(X, y, lwState.tau);

  document.getElementById('lw-rmse').textContent = lwlrRMSE(X, y, lwState.tau);
  document.getElementById('lw-tau').textContent = lwState.tau.toFixed(2);
  document.getElementById('lw-sigma').textContent = lwState.sigma.toFixed(2);

  const datasets = [
    {
      type: 'scatter',
      label: 'sin(x) + noise',
      data: X.map((xi, i) => ({ x: xi, y: y[i] })),
      backgroundColor: 'rgba(244,114,182,0.55)',
      borderColor: 'rgba(244,114,182,0.2)',
      borderWidth: 1,
      pointRadius: 5,
      pointHoverRadius: 7
    },
    {
      type: 'line',
      label: `LWLR (τ=${lwState.tau.toFixed(2)})`,
      data: Xt.map((xi, i) => ({ x: xi, y: yp[i] })),
      borderColor: '#fb923c',
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.3,
      fill: false
    }
  ];

  if (lwState.showTrue) {
    const Xt2 = Array.from({ length: 300 }, (_, i) => -3 + 6 * i / 299);
    datasets.push({
      type: 'line',
      label: 'True sin(x)',
      data: Xt2.map(xi => ({ x: xi, y: Math.sin(xi) })),
      borderColor: 'rgba(148,163,184,0.5)',
      borderWidth: 1.5,
      borderDash: [6, 4],
      pointRadius: 0,
      tension: 0.4,
      fill: false
    });
  }

  if (!lwChart) {
    const ctx = document.getElementById('lwlrChart').getContext('2d');
    lwChart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: chartOptions('x', 'y = sin(x) + noise', '#f472b6')
    });
  } else {
    lwChart.data.datasets = datasets;
    lwChart.update('active');
  }
}

// ── Shared chart options builder ─────────────────────
function chartOptions(xLabel, yLabel, accentColor) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400, easing: 'easeOutQuart' },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        align: 'end',
        labels: {
          color: '#8080a8',
          font: { size: 11, weight: '500' },
          usePointStyle: true,
          pointStyleWidth: 8,
          padding: 16
        }
      },
      tooltip: {
        backgroundColor: 'rgba(4,4,16,0.95)',
        borderColor: `${accentColor}44`,
        borderWidth: 1,
        titleColor: '#f0f0ff',
        bodyColor: '#8080a8',
        padding: 10,
        cornerRadius: 10,
        callbacks: {
          label: item => `${item.dataset.label}: (${item.parsed.x.toFixed(2)}, ${item.parsed.y.toFixed(2)})`
        }
      }
    },
    scales: {
      x: {
        title: { display: true, text: xLabel, color: '#8080a8', font: { size: 11 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#3a3a56', font: { family: 'JetBrains Mono', size: 9 } }
      },
      y: {
        title: { display: true, text: yLabel, color: '#8080a8', font: { size: 11 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#3a3a56', font: { family: 'JetBrains Mono', size: 9 } }
      }
    }
  };
}

// ═══════════════════════════════════════════════════
//  TAB SYSTEM
// ═══════════════════════════════════════════════════
function setupTabs() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      if (tab === activeTab) return;
      activeTab = tab;

      // Update tab buttons
      document.querySelectorAll('.tab').forEach(t => {
        t.classList.remove('active', 'tab-lr', 'tab-pr', 'tab-lw');
      });
      btn.classList.add('active', tab === 'lr' ? 'tab-lr' : tab === 'pr' ? 'tab-pr' : 'tab-lw');

      // Update panels
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.getElementById(`panel-${tab}`).classList.add('active');

      // Lazy-init charts
      if (tab === 'lr' && !lrChart) initLRChart();
      if (tab === 'pr' && !prChart) buildPRChart(polyDegree);
      if (tab === 'lwlr' && !lwChart) buildLWChart();

      // Resize fix
      setTimeout(() => {
        if (tab === 'lr' && lrChart) lrChart.resize();
        if (tab === 'pr' && prChart) prChart.resize();
        if (tab === 'lwlr' && lwChart) lwChart.resize();
      }, 50);
    });
  });

  // Set initial active class
  document.getElementById('tab-lr').classList.add('active', 'tab-lr');
}

// ═══════════════════════════════════════════════════
//  CONTROLS
// ═══════════════════════════════════════════════════
function setupPRControls() {
  document.querySelectorAll('.deg-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      polyDegree = parseInt(btn.dataset.deg);
      document.querySelectorAll('.deg-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      buildPRChart(polyDegree);
    });
  });
}

function setupLWControls() {
  const tauSlider = document.getElementById('lwTau');
  const sigmaSlider = document.getElementById('lwSigma');

  tauSlider.addEventListener('input', () => {
    lwState.tau = parseFloat(tauSlider.value);
    document.getElementById('lwTauVal').textContent = lwState.tau.toFixed(2);
    buildLWChart();
  });

  sigmaSlider.addEventListener('input', () => {
    lwState.sigma = parseFloat(sigmaSlider.value);
    document.getElementById('lwSigmaVal').textContent = lwState.sigma.toFixed(2);
    lwData = generateLwlrData(lwState.sigma);
    buildLWChart();
  });

  document.getElementById('lwRegen').addEventListener('click', () => {
    lwlrSeed = Math.floor(Math.random() * 99999);
    lwData = generateLwlrData(lwState.sigma);
    buildLWChart();
  });

  document.getElementById('lwTruth').addEventListener('click', () => {
    lwState.showTrue = !lwState.showTrue;
    document.getElementById('lwTruth').classList.toggle('on', lwState.showTrue);
    buildLWChart();
  });

  document.getElementById('lwReset').addEventListener('click', () => {
    lwState = { tau: 0.5, sigma: 0.2, showTrue: false };
    tauSlider.value = 0.5; sigmaSlider.value = 0.2;
    document.getElementById('lwTauVal').textContent = '0.50';
    document.getElementById('lwSigmaVal').textContent = '0.20';
    document.getElementById('lwTruth').classList.remove('on');
    resetLwlrSeed(42);
    lwData = generateLwlrData(0.2);
    buildLWChart();
  });
}

// ═══════════════════════════════════════════════════
//  NAV HIGHLIGHT
// ═══════════════════════════════════════════════════
function setupNav() {
  const sections = ['hero','models-section','theory-section','code-section'];
  const navMap = { 'hero':'nl-home','models-section':'nl-models','theory-section':'nl-theory','code-section':'nl-code' };
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        const el = document.getElementById(navMap[e.target.id]);
        if (el) el.classList.add('active');
      }
    });
  }, { threshold: 0.4 });
  sections.forEach(id => { const el = document.getElementById(id); if (el) obs.observe(el); });
}

// ═══════════════════════════════════════════════════
//  COPY CODE
// ═══════════════════════════════════════════════════
function copyCode() {
  const code = `import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Linear Regression
h = fetch_california_housing()
X = pd.DataFrame(h.data, columns=h.feature_names)[['MedInc']]
y = h.target
x1,x2,y1,y2 = train_test_split(X,y,test_size=0.2,random_state=42)
m = LinearRegression().fit(x1,y1)
plt.scatter(x2,y2); plt.plot(x2,m.predict(x2)); plt.show()

# Polynomial Regression
d = sns.load_dataset('mpg').dropna()
X,y = d[['horsepower']], d['mpg']
p = PolynomialFeatures(2)
m.fit(p.fit_transform(X), y)
x = np.linspace(X.min(),X.max(),100).reshape(-1,1)
plt.scatter(X,y); plt.plot(x,m.predict(p.transform(x))); plt.show()

# LWLR
X = np.linspace(-3,3,100).reshape(-1,1)
y = np.sin(X).ravel()+np.random.normal(0,0.2,100)

def f(x0,t=0.5):
    w=np.exp(-np.sum((X-x0)**2,1)/(2*t*t))
    W=np.diag(w); X1=np.c_[np.ones(len(X)),X]
    th=np.linalg.pinv(X1.T@W@X1)@X1.T@W@y
    return np.r_[1,x0]@th

xt=np.linspace(-3,3,300).reshape(-1,1)
plt.scatter(X,y); plt.plot(xt,[f(i) for i in xt]); plt.show()`;

  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✓ Copied!';
    btn.style.color = '#4ade80';
    setTimeout(() => { btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy'; btn.style.color = ''; }, 2000);
  });
}

// ═══════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupPRControls();
  setupLWControls();
  setupNav();
  // Init the first tab chart
  initLRChart();
});
