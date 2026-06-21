# 📊 ML Regression Visualizer

<div align="center">

![ML Regression](https://img.shields.io/badge/ML-Regression%20Visualizer-818cf8?style=for-the-badge&logo=python&logoColor=white)
![Linear](https://img.shields.io/badge/Linear%20Regression-California%20Housing-38bdf8?style=for-the-badge)
![Polynomial](https://img.shields.io/badge/Polynomial%20Regression-MPG%20Dataset-a78bfa?style=for-the-badge)
![LWLR](https://img.shields.io/badge/LWLR-sin(x)%20%2B%20Noise-f472b6?style=for-the-badge)
![GitHub Pages](https://img.shields.io/badge/GitHub-deepthi--tr05-181717?style=for-the-badge&logo=github&logoColor=white)
![Vercel](https://img.shields.io/badge/Deploy-Vercel%20Ready-000000?style=for-the-badge&logo=vercel&logoColor=white)

**Three regression models in one stunning interactive dashboard.**  
Explore **Linear Regression**, **Polynomial Regression**, and **Locally Weighted Linear Regression (LWLR)** — all implemented from scratch in JavaScript with real-time controls.

[🚀 Live Demo (Vercel)](https://ml-regression-viz.vercel.app) &nbsp;|&nbsp; [📁 Repository](https://github.com/deepthi-tr05/ml-regression-viz) &nbsp;|&nbsp; [📊 View Visualizer](https://github.com/deepthi-tr05/ml-regression-viz#models)

</div>

---

## 🎯 Overview

This project provides an **interactive web-based visualization** for three fundamental regression techniques used in Machine Learning:

| # | Model | Dataset | Key Parameter |
|---|---|---|---|
| 01 | 📈 Linear Regression | California Housing (sklearn) | OLS — no hyperparameters |
| 02 | 🎡 Polynomial Regression | Auto MPG (seaborn) | Degree `d` (1 to 10) |
| 03 | 🔔 LWLR | sin(x) + Gaussian noise | Bandwidth `τ` (0.1 to 2.0) |

---

## ✨ Interactive Features

- 🔀 **Tab switcher** — jump between all 3 models instantly
- 🎛️ **Polynomial degree selector** — buttons for degree 1, 2, 3, 4, 6, 10 with live curve update
- 🔊 **LWLR bandwidth τ slider** — drag to shift between overfitting and underfitting
- 📉 **Noise σ slider** — control Gaussian noise on LWLR training data  
- 🔄 **Regenerate button** — fresh random dataset each click
- 👁️ **True sin(x) overlay** — compare LWR fit to ground truth
- 📐 **Live metrics** — R² and RMSE updated in real-time for every change
- 📋 **One-click copy** — copy full Python source code to clipboard
- 📊 **Model comparison table** — theory side-by-side
- 🌌 **Animated mesh background** with floating gradient orbs

---

## 🖼️ App Sections

```
Hero         → 3 model pills, CTA buttons, animated gradient title
Models       → Tabbed charts with controls for each regression model
Theory       → Comparison table + formula cards for all 3 models
Code         → Styled Python source code with copy button
```

---

## 🔬 Model Details

### 1. 📈 Linear Regression — California Housing

Uses **Ordinary Least Squares** to predict median house value from median income.

```
ŷ  =  θ₀ + θ₁ · MedInc
θ  =  (XᵀX)⁻¹ Xᵀy
```

- **Dataset:** 20,640 samples · 80/20 train/test split · `random_state=42`
- **Feature:** `MedInc` (median income in $10k blocks)
- **Target:** Median house value in $100k units

---

### 2. 🎡 Polynomial Regression — MPG Dataset

Maps features to a **d-dimensional polynomial space**, then applies linear regression.

```
φ(x) = [1,  x,  x²,  …,  xᵈ]
ŷ    =  θᵀ · φ(x)
```

| Degree | R² (approx) | Behaviour |
|---|---|---|
| 1 | ~0.60 | Underfitting |
| 2 | ~0.69 | Good fit |
| 3 | ~0.71 | Slightly better |
| 10 | ⚠️ | Overfitting |

---

### 3. 🔔 LWLR — sin(x) + Gaussian Noise

Fits a **local weighted linear model** at every query point using a Gaussian kernel.

```
w(i)  =  exp(−‖xᵢ − x₀‖²  /  2τ²)     ← kernel weight
θ     =  (XᵀWX)⁻¹ XᵀWy                 ← weighted normal equations
ŷ(x₀) =  [1, x₀]ᵀ · θ                  ← local prediction
```

| τ value | Effect | Bias | Variance |
|---|---|---|---|
| τ < 0.2 | Overfitting | Low | **High** |
| τ ≈ 0.5 | **Optimal** | Balanced | Balanced |
| τ > 1.5 | Underfitting | **High** | Low |

---

## 🐍 Python Source Code

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# ── 1. Linear Regression ──────────────────────────────
h = fetch_california_housing()
X = pd.DataFrame(h.data, columns=h.feature_names)[['MedInc']]
y = h.target
x1,x2,y1,y2 = train_test_split(X, y, test_size=0.2, random_state=42)
m = LinearRegression().fit(x1, y1)
plt.scatter(x2, y2)
plt.plot(x2, m.predict(x2))
plt.show()

# ── 2. Polynomial Regression ──────────────────────────
d = sns.load_dataset('mpg').dropna()
X, y = d[['horsepower']], d['mpg']
p = PolynomialFeatures(2)
m.fit(p.fit_transform(X), y)
x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.scatter(X, y)
plt.plot(x, m.predict(p.transform(x)))
plt.show()

# ── 3. LWLR ───────────────────────────────────────────
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 100)

def f(x0, t=0.5):
    w  = np.exp(-np.sum((X - x0)**2, 1) / (2 * t * t))
    W  = np.diag(w)
    X1 = np.c_[np.ones(len(X)), X]
    th = np.linalg.pinv(X1.T @ W @ X1) @ X1.T @ W @ y
    return np.r_[1, x0] @ th

xt = np.linspace(-3, 3, 300).reshape(-1, 1)
plt.scatter(X, y)
plt.plot(xt, [f(i) for i in xt])
plt.show()
```

---

## 🚀 Run Python Script Locally

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Run all 3 models
python regression_models.py
```

---

## 🌐 Deploy Live Demo

This project includes `vercel.json` for zero-config static deployment:

```bash
# Option 1: Vercel (recommended)
# 1. Go to https://vercel.com/new
# 2. Import github.com/deepthi-tr05/ml-regression-viz
# 3. Click Deploy → done in ~30 seconds

# Option 2: GitHub Pages
# Settings → Pages → Branch: main → / (root) → Save
```

---

## 📁 Project Structure

```
ml-regression-viz/
├── index.html              # Full web app with tabs, controls, theory, code
├── style.css               # Dark glassmorphism design + animations
├── app.js                  # All 3 models implemented from scratch in JS
├── data.js                 # Embedded datasets (Housing, MPG) + seeded RNG
├── regression_models.py    # Original Python script (all 3 models)
├── requirements.txt        # Python dependencies
├── vercel.json             # Static site config (no Python error on Vercel)
└── README.md               # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Structure | HTML5 (Semantic) |
| Styling | Vanilla CSS3 (Glassmorphism, Mesh Orbs, Animations) |
| Interactivity | Vanilla JavaScript ES6+ |
| Charts | Chart.js 4.4 |
| Algorithms | Linear Regression, Polynomial Regression, LWLR — all in JS |
| Python ML | NumPy · Pandas · Matplotlib · Seaborn · scikit-learn |
| Fonts | Inter + JetBrains Mono (Google Fonts) |
| Hosting | Vercel / GitHub Pages |

---

## 📊 Comparison Summary

| Property | Linear | Polynomial | LWLR |
|---|---|---|---|
| Type | Parametric | Parametric | Non-parametric |
| Curve shape | Straight line | Curved | Locally adaptive |
| Key hyperparameter | — | Degree `d` | Bandwidth `τ` |
| Overfitting risk | Low | High (large d) | High (small τ) |
| Training cost | O(n) | O(n·d) | O(n²) per query |
| Dataset | California Housing | Auto MPG | sin(x) + noise |

---

## 👩‍💻 Author

**Deepthi** — [@deepthi-tr05](https://github.com/deepthi-tr05)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Made with ❤️ using HTML · CSS · JavaScript · Chart.js  
Data: California Housing (sklearn) · Auto MPG (seaborn) · sin(x)+noise

**[⭐ Star this repo](https://github.com/deepthi-tr05/ml-regression-viz) if you found it useful!**

</div>
