# 🌳 Introduction to Decision Trees

> **Course:** Big Data and Machine Learning
> **Author / Instructor:** Abel Tadesse ([ab3lT](https://github.com/ab3lT))
>
> Hands-on, fully-executed Jupyter notebooks that teach **Decision Trees** for **classification** and **regression** —
> with the concepts, the mathematics, and working scikit-learn code side by side.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)

---

## ⭐ Students: Star & Fork this repository!

1. Click **⭐ Star** (top-right of this page) so you can find the material again easily.
2. Click **🍴 Fork** to get your **own copy** — that is where you will continue the work and solve the exercises.
3. Clone *your fork*, run the notebooks, experiment, commit and push your progress.

```bash
git clone https://github.com/<your-username>/decision-trees-big-data-ml.git
cd decision-trees-big-data-ml
```

---

## 📂 Repository structure

```
decision-trees-big-data-ml/
├── 01_decision_tree_classification.ipynb   # 🏦 Bank Marketing -> will the customer subscribe?
├── 02_decision_tree_regression.ipynb       # 🚗 Car prices     -> predict the MSRP
├── content/
│   ├── bank_marketing.csv                  # dataset for classification
│   └── Car Price Prediction.csv            # dataset for regression
├── decision_tree_classification.pdf        # rendered classification tree
├── decision_tree_regression.pdf            # rendered regression tree
├── requirements.txt
└── README.md
```

## 📓 Notebooks

| # | Notebook | Task | Dataset | Open in Colab |
|---|----------|------|---------|---------------|
| 1 | [Decision Tree Classification](01_decision_tree_classification.ipynb) | Predict if a customer subscribes to a term deposit | `bank_marketing.csv` (11,162 rows) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ab3lT/decision-trees-big-data-ml/blob/main/01_decision_tree_classification.ipynb) |
| 2 | [Decision Tree Regression](02_decision_tree_regression.ipynb) | Predict a car's price (MSRP) | `Car Price Prediction.csv` (11,914 rows) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ab3lT/decision-trees-big-data-ml/blob/main/02_decision_tree_regression.ipynb) |

All cells are **already executed**, so you can read the outputs and plots directly on GitHub.

### What each notebook covers
1. **Concepts** – root, internal nodes, leaves, depth, the CART greedy algorithm
2. **Mathematics** – impurity measures, split criterion, worked examples (verified in code)
3. **Exploratory data analysis** – schema, distinct values, summary statistics, target distribution
4. **Cleaning & feature engineering** – missing/`unknown` values, `Car_Age`, one-hot encoding, scaling
5. **Pipeline** – `ColumnTransformer` + `DecisionTree*` in a scikit-learn `Pipeline`
6. **Evaluation** – metrics computed with scikit-learn **and by hand** from their formulas
7. **Visualization** – `plot_tree`, text rules, the step-function view of regression trees
8. **Feature importance** – mean impurity decrease
9. **Overfitting** – train vs. test score as `max_depth` grows (bias–variance trade-off)
10. **Tuning** – `GridSearchCV` with k-fold cross-validation
11. **Exercises** – to continue in your fork

---

## 🧮 Key mathematics (cheat sheet)

For a node with class proportions $p_k$:

| Measure | Formula | Used by |
|---|---|---|
| Gini impurity | $G = 1 - \sum_k p_k^2$ | Classification (`criterion='gini'`) |
| Entropy | $H = -\sum_k p_k \log_2 p_k$ | Classification (`criterion='entropy'`) |
| MSE (variance) | $\text{MSE}(t) = \frac{1}{N_t}\sum_{i\in t}(y_i - \bar y_t)^2$ | Regression (`criterion='squared_error'`) |

**Best split** – choose feature $j$ and threshold $t$ that maximize the impurity decrease:

$$
\Delta I = I(P) - \left( \frac{N_L}{N_P} I(L) + \frac{N_R}{N_P} I(R) \right)
$$

**Leaf prediction** – majority class (classification) or mean $\bar y$ (regression).

**Classification metrics**

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad
\text{Precision}=\frac{TP}{TP+FP},\quad
\text{Recall}=\frac{TP}{TP+FN},\quad
F_1=\frac{2\,PR}{P+R}
$$

**Regression metrics**

$$
\text{MAE}=\frac{1}{n}\sum|y_i-\hat y_i|,\quad
\text{RMSE}=\sqrt{\frac{1}{n}\sum(y_i-\hat y_i)^2},\quad
R^2 = 1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}
$$

---

## 🚀 Getting started

### Option A – Google Colab (no installation)
Click an **Open in Colab** badge above. The datasets are downloaded automatically from this repository.

### Option B – Run locally
```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2. install the dependencies
pip install -r requirements.txt

# 3. launch Jupyter and open the notebooks
jupyter notebook
```

---

## ✍️ Exercises for students

Each notebook ends with a list of exercises. Some highlights:

- Remove the `duration` feature (it leaks the outcome) and compare the results.
- Switch the criterion to `entropy` / `absolute_error` and verify the math by hand.
- Post-prune the tree with **cost-complexity pruning** (`ccp_alpha`).
- Train on `log(MSRP)` to handle the skewed price distribution.
- Compare a single tree with **Random Forests** and **Gradient Boosting**.

**Workflow:** fork ➜ create a branch (`git checkout -b exercises`) ➜ solve ➜ commit ➜ push to your fork.
Found a bug or have an improvement? Open an **Issue** or a **Pull Request**!

---

## 📚 Further reading
- [scikit-learn User Guide – Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- Breiman, Friedman, Olshen & Stone (1984), *Classification and Regression Trees* (CART)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* – Chapter 8: Tree-Based Methods

---

## 👤 Author

**Abel Tadesse** — prepared for the *Big Data and Machine Learning* course.

- GitHub: [ab3lT](https://github.com/ab3lT)

If this material helped you, please **⭐ star** the repository — it helps other students find it!
