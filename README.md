# 📊 ML Regression Visualizer

<div align="center">

![ML Regression](https://img.shields.io/badge/ML-Regression%20Visualizer-818cf8?style=for-the-badge&logo=python&logoColor=white)
![Linear](https://img.shields.io/badge/Linear%20Regression-California%20Housing-38bdf8?style=for-the-badge)
![Polynomial](https://img.shields.io/badge/Polynomial%20Regression-MPG%20Dataset-a78bfa?style=for-the-badge)
![LWLR](https://img.shields.io/badge/LWLR-sin(x)%20%2B%20Noise-f472b6?style=for-the-badge)
![Vercel](https://img.shields.io/badge/Live%20Demo-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)

**Three regression models in one interactive dashboard.**  
Linear Regression · Polynomial Regression · Locally Weighted Linear Regression

[🚀 Live Demo](https://ml-regression-viz.vercel.app) • [📊 Models](https://ml-regression-viz.vercel.app/#models-section) • [💻 Code](https://github.com/deepthi-tr05/ml-regression-viz)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📈 **Linear Regression** | California Housing — MedInc vs House Value with R² & RMSE |
| 🎡 **Polynomial Regression** | MPG Dataset — degree selector (1, 2, 3, 4, 6, 10) with live curve |
| 🔔 **LWLR** | sin(x)+noise — bandwidth τ & noise σ sliders, live RMSE |
| 🔢 **Live Metrics** | R², RMSE updated in real-time for every change |
| 🔄 **Regenerate Data** | Fresh random datasets on click |
| 👁️ **True Curve Toggle** | Overlay true sin(x) on LWLR chart |
| 📋 **Copy Code** | One-click copy of full Python source |
| 📐 **Model Comparison Table** | Side-by-side theory comparison of all 3 models |

---

## 🌐 Live Demo

👉 **[https://ml-regression-viz.vercel.app](https://ml-regression-viz.vercel.app)**

---

## 🔬 Models Explained

### 1. 📈 Linear Regression — California Housing

Predicts **median house value** from **median income** using Ordinary Least Squares.

```
ŷ = θ₀ + θ₁ · MedInc
θ = (XᵀX)⁻¹ Xᵀy
```

**Dataset:** sklearn's California Housing (20,640 samples, 80/20 split)

---

### 2. 🎡 Polynomial Regression — MPG Dataset

Fits a **degree-d polynomial** to horsepower vs miles-per-gallon.

```
φ(x) = [1, x, x², …, xᵈ]
ŷ = θ · φ(x)
```

| Degree | R² | RMSE |
|---|---|---|
| 1 | ~0.60 | ~4.92 |
| 2 | ~0.69 | ~4.40 |
| 3 | ~0.71 | ~4.28 |
| 10 | ⚠️ Overfits | High |

---

### 3. 🔔 LWLR — sin(x) + Noise

Locally Weighted Linear Regression fits a **separate model at every query point** using Gaussian kernel weights.

```
w(i) = exp(−‖xᵢ − x₀‖² / 2τ²)
θ    = (XᵀWX)⁻¹ XᵀWy
```

| τ | Effect |
|---|---|
| < 0.2 | Overfitting |
| ≈ 0.5 | Optimal |
| > 1.5 | Underfitting |

---

## 🐍 Python Source Code

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
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

def f(x0, t=0.5):
    w  = np.exp(-np.sum((X-x0)**2,1)/(2*t*t))
    W  = np.diag(w); X1 = np.c_[np.ones(len(X)),X]
    th = np.linalg.pinv(X1.T@W@X1)@X1.T@W@y
    return np.r_[1,x0]@th

xt = np.linspace(-3,3,300).reshape(-1,1)
plt.scatter(X,y); plt.plot(xt,[f(i) for i in xt]); plt.show()
```

---

## 🚀 Run Locally

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python regression_models.py
```

---

## 📁 Project Structure

```
ml-regression-viz/
├── index.html              # Tabbed web app
├── style.css               # Dark glassmorphism design
├── app.js                  # All 3 models in JavaScript
├── data.js                 # Embedded datasets + RNG
├── regression_models.py    # Python source script
├── requirements.txt        # Python dependencies
├── vercel.json             # Vercel static config
└── README.md               # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Structure | HTML5 |
| Styling | Vanilla CSS3 (Glassmorphism) |
| Interactivity | Vanilla JavaScript ES6+ |
| Charts | Chart.js 4.4 |
| Python ML | NumPy, Pandas, scikit-learn, Seaborn |
| Fonts | Inter + JetBrains Mono |
| Hosting | Vercel |

---

## 👩‍💻 Author

**Deepthi** — [@deepthi-tr05](https://github.com/deepthi-tr05)

---

## 📄 License

MIT License
