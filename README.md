# XGBoost Classification From Scratch

## Overview

This project demonstrates how XGBoost works internally by implementing the core concepts of binary classification step by step before comparing the results with the official XGBoost library.

The implementation covers:

* Log-Odds Initialization
* Sigmoid Transformation
* Gradient Calculation
* Hessian Calculation
* Gain-Based Split Selection
* Leaf Weight Computation
* Prediction Updates
* Decision Boundary Visualization
* XGBoost Classifier using the official library
* Hyperparameter Tuning with GridSearchCV

The dataset used is the Moon Dataset generated using `sklearn.datasets.make_moons`.

---

## Dataset

```python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=500,
    noise=0.25,
    random_state=42
)
```

The dataset contains two features, making it ideal for visualizing decision boundaries.

---

## XGBoost From Scratch

### Step 1: Base Prediction

Compute the initial prediction using log-odds:

[
\text{Base Score} = \log\left(\frac{p}{1-p}\right)
]

where:

[
p = \text{mean}(y)
]

---

### Step 2: Probability Prediction

Apply the sigmoid function:

[
\sigma(x)=\frac{1}{1+e^{-x}}
]

---

### Step 3: Gradient Calculation

[
g_i = p_i - y_i
]

---

### Step 4: Hessian Calculation

[
h_i = p_i(1-p_i)
]

---

### Step 5: Gain Calculation

[
Gain=
\frac{1}{2}
\left(
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-------------------------

\frac{G^2}{H+\lambda}
\right)
]

This gain metric is used to find the best split.

---

### Step 6: Leaf Weight

[
w^*=
-\frac{\sum g_i}
{\sum h_i+\lambda}
]

---

### Step 7: Prediction Update

[
F_{new}(x)=
F_{old}(x)
+
\eta \times Tree(x)
]

where:

* η = learning rate

---

## Official XGBoost Implementation

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

## Hyperparameter Tuning

GridSearchCV was used to find the best model.

```python
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [2,3,4,5],
    'learning_rate': [0.01,0.05,0.1,0.2],
    'subsample': [0.8,1.0]
}
```

---

## Visualizations

* Moon Dataset Visualization
* Best Split Visualization
* First Tree Prediction
* Decision Boundary
* Decision Boundary for Different Tree Depths
* Decision Boundary of the Best GridSearchCV Model

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* XGBoost

---

---

## Key Learnings

* Difference between Gradient Boosting and XGBoost
* Role of Gradients and Hessians
* Gain-based splitting
* Leaf weight optimization
* Effect of hyperparameters on decision boundaries
* Practical implementation of XGBoost using Scikit-Learn

---

## Author

Raj Kumar Gupta

B.Tech CSE (AI & ML)
Amity University Jharkhand
