# 🚀 XGBoost Classification From Scratch

## 📌 Overview

This project explores how XGBoost works internally by implementing the core concepts of binary classification from scratch and then comparing the results with the official XGBoost library.

The project covers:

* ✅ Log-Odds Initialization
* ✅ Sigmoid Function
* ✅ Gradient Calculation
* ✅ Hessian Calculation
* ✅ Gain-Based Split Selection
* ✅ Leaf Weight Computation
* ✅ Prediction Updates
* ✅ Decision Boundary Visualization
* ✅ XGBoost Classifier
* ✅ Hyperparameter Tuning with GridSearchCV

---

## 📊 Dataset

The Moon Dataset was generated using Scikit-Learn.

```python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=500,
    noise=0.25,
    random_state=42
)
```

This dataset contains two features, making it ideal for visualizing decision boundaries.

---

## 🧠 XGBoost From Scratch

### 1️⃣ Base Prediction (Log Odds)

The initial prediction is calculated as:

```math
p = \frac{\text{Number of Positive Samples}}{\text{Total Samples}}
```

```math
F_0 = \log\left(\frac{p}{1-p}\right)
```

---

### 2️⃣ Sigmoid Function

Convert log-odds into probabilities:

```math
\sigma(x)=\frac{1}{1+e^{-x}}
```

---

### 3️⃣ Gradient Calculation

For binary log-loss:

```math
g_i = p_i - y_i
```

where:

* $p_i$ = predicted probability
* $y_i$ = actual label

---

### 4️⃣ Hessian Calculation

```math
h_i = p_i(1-p_i)
```

---

### 5️⃣ Gain Calculation

XGBoost chooses the split with the maximum gain:

```math
Gain=
\frac{1}{2}
\left(
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{G^2}{H+\lambda}
\right)
```

where:

* $G_L, G_R$ = Sum of gradients
* $H_L, H_R$ = Sum of hessians
* $\lambda$ = Regularization parameter

---

### 6️⃣ Optimal Leaf Weight

The prediction value assigned to a leaf is:

```math
w^* =
-\frac{\sum g_i}
{\sum h_i + \lambda}
```

---

### 7️⃣ Prediction Update

Each boosting round updates the model prediction:

```math
F_{m}(x)
=
F_{m-1}(x)
+
\eta \cdot T_m(x)
```

where:

* $\eta$ = Learning Rate
* $T_m(x)$ = Prediction from the current tree

---

## 🌲 XGBoost Using Library

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
```

---

## 🔍 Hyperparameter Tuning

GridSearchCV was used to find the optimal parameters.

```python
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [2,3,4,5],
    'learning_rate': [0.01,0.05,0.1,0.2],
    'subsample': [0.8,1.0]
}
```

Best Parameters:

```python
{
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 1.0
}
```

---

## 📈 Visualizations

* Moon Dataset Visualization
* Best Split Visualization
* First Tree Prediction
* Decision Boundary
* Probability Surface
* GridSearchCV Best Model Decision Boundary
* Decision Boundary Comparison for Different Tree Depths

---

## 🛠️ Technologies Used

<div align="left">

<img src="https://skillicons.dev/icons?i=python" height="40"/>
<img src="https://skillicons.dev/icons?i=numpy" height="40"/>
<img src="https://skillicons.dev/icons?i=pandas" height="40"/>

</div>

* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* XGBoost

---

## 🎯 Key Learnings

* Understanding XGBoost Internals
* Role of Gradients and Hessians
* Gain-Based Tree Construction
* Leaf Weight Optimization
* Effect of Hyperparameters on Decision Boundaries
* Comparison Between Manual and Library Implementations

---

## 👨‍💻 Author

**Raj Kumar Gupta**

B.Tech CSE (AI & ML)
Amity University Jharkhand
