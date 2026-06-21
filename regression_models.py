import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Linear Regression ─────────────────────────────────
print("=" * 50)
print("1. LINEAR REGRESSION — California Housing")
print("=" * 50)
h = fetch_california_housing()
X = pd.DataFrame(h.data, columns=h.feature_names)[['MedInc']]
y = h.target
x1, x2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=42)

m = LinearRegression().fit(x1, y1)
pred = m.predict(x2)
print(f"  Coefficient : {m.coef_[0]:.4f}")
print(f"  Intercept   : {m.intercept_:.4f}")
print(f"  R² Score    : {r2_score(y2, pred):.4f}")
print(f"  RMSE        : {np.sqrt(mean_squared_error(y2, pred)):.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#040410'); ax.set_facecolor('#090918')
x_sorted = x2.sort_values('MedInc')
ax.scatter(x2, y2, color='#38bdf8', alpha=0.4, s=20, label='Test data')
ax.plot(x_sorted, m.predict(x_sorted), color='#0ea5e9', linewidth=2.5, label='Linear fit')
ax.set_xlabel('Median Income', color='#8080a8'); ax.set_ylabel('House Value ($100k)', color='#8080a8')
ax.set_title('Linear Regression — California Housing', color='#f0f0ff', fontweight='bold')
ax.legend(labelcolor='#ccccdd', facecolor='#090918', edgecolor='#333355', framealpha=0.3)
ax.tick_params(colors='#3a3a56')
for s in ax.spines.values(): s.set_edgecolor('#1e1e3a')
ax.grid(alpha=0.06, color='white')
plt.tight_layout(); plt.savefig('linear_regression.png', dpi=150, facecolor='#040410'); plt.show()

# ── 2. Polynomial Regression ─────────────────────────────
print("\n" + "=" * 50)
print("2. POLYNOMIAL REGRESSION — MPG Dataset")
print("=" * 50)
d = sns.load_dataset('mpg').dropna()
X, y = d[['horsepower']], d['mpg']

for degree in [1, 2, 3]:
    p = PolynomialFeatures(degree)
    Xp = p.fit_transform(X)
    m2 = LinearRegression().fit(Xp, y)
    pred2 = m2.predict(Xp)
    print(f"  Degree {degree}: R²={r2_score(y, pred2):.4f}  RMSE={np.sqrt(mean_squared_error(y, pred2)):.4f}")

# Plot degree=2
p = PolynomialFeatures(2)
m2 = LinearRegression().fit(p.fit_transform(X), y)
x_curve = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#040410'); ax.set_facecolor('#090918')
ax.scatter(X, y, color='#818cf8', alpha=0.5, s=25, label='Data')
ax.plot(x_curve, m2.predict(p.transform(x_curve)), color='#a78bfa', linewidth=2.5, label='Poly degree 2')
ax.set_xlabel('Horsepower', color='#8080a8'); ax.set_ylabel('MPG', color='#8080a8')
ax.set_title('Polynomial Regression — MPG Dataset', color='#f0f0ff', fontweight='bold')
ax.legend(labelcolor='#ccccdd', facecolor='#090918', edgecolor='#333355', framealpha=0.3)
ax.tick_params(colors='#3a3a56')
for s in ax.spines.values(): s.set_edgecolor('#1e1e3a')
ax.grid(alpha=0.06, color='white')
plt.tight_layout(); plt.savefig('poly_regression.png', dpi=150, facecolor='#040410'); plt.show()

# ── 3. Locally Weighted Linear Regression (LWLR) ─────────
print("\n" + "=" * 50)
print("3. LOCALLY WEIGHTED LINEAR REGRESSION (LWLR)")
print("=" * 50)
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 100)

def f(x0, t=0.5):
    w  = np.exp(-np.sum((X - x0)**2, 1) / (2 * t * t))
    W  = np.diag(w)
    X1 = np.c_[np.ones(len(X)), X]
    th = np.linalg.pinv(X1.T @ W @ X1) @ X1.T @ W @ y
    return np.r_[1, x0] @ th

xt = np.linspace(-3, 3, 300).reshape(-1, 1)
yp = np.array([f(i) for i in xt])
rmse_lw = np.sqrt(np.mean((yp - np.sin(xt.ravel()))**2))
print(f"  RMSE vs sin(x): {rmse_lw:.4f}  (tau=0.5)")

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#040410'); ax.set_facecolor('#090918')
ax.scatter(X, y, color='#f472b6', alpha=0.6, s=25, label='sin(x) + noise')
ax.plot(xt, yp, color='#fb923c', linewidth=2.5, label='LWLR fit (τ=0.5)')
ax.plot(xt, np.sin(xt), color='#94a3b8', linewidth=1.2, linestyle='--', alpha=0.5, label='True sin(x)')
ax.set_xlabel('x', color='#8080a8'); ax.set_ylabel('y', color='#8080a8')
ax.set_title('LWLR — sin(x) + Gaussian Noise', color='#f0f0ff', fontweight='bold')
ax.legend(labelcolor='#ccccdd', facecolor='#090918', edgecolor='#333355', framealpha=0.3)
ax.tick_params(colors='#3a3a56')
for s in ax.spines.values(): s.set_edgecolor('#1e1e3a')
ax.grid(alpha=0.06, color='white')
plt.tight_layout(); plt.savefig('lwlr.png', dpi=150, facecolor='#040410'); plt.show()
print("\nAll plots saved!")
