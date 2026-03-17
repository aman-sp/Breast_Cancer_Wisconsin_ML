# Breast Cancer Wisconsin — Full ML Pipeline

> This notebook presents a full machine learning pipeline on the Breast Cancer Wisconsin dataset. It covers EDA with feature distributions, correlation heatmaps, and outlier detection, followed by PCA and MDS. Four supervised models are trained and evaluated alongside K-Means and hierarchical clustering for unsupervised analysis.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Dataset

| Property | Value |
|---|---|
| Source | `sklearn.datasets.load_breast_cancer` |
| Samples | 569 |
| Features | 30 numeric (mean, error, worst) |
| Classes | Malignant (212) · Benign (357) |
| Download | Not required — built into scikit-learn |

---

## Notebook Structure

### 1. Exploratory Data Analysis (EDA)
- Class distribution bar chart
- Feature histograms split by Malignant vs Benign
- Correlation heatmap (mean features + target)
- Box plots & IQR-based outlier detection per feature

### 2. Dimensionality Reduction
- **PCA** — 2D scatter plot + cumulative explained variance curve (95% threshold)
- **MDS** — 2D visualization on a 300-sample subset

### 3. Supervised Models
| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` |
| MLP Classifier | Hidden layers: `128 → 64` |
| SVM | RBF kernel, `C=1.0`, `gamma=scale` |
| Random Forest | `n_estimators=200` (Ensemble) |

### 4. Unsupervised Learning
- **K-Means** — Elbow method (k=2–10) + cluster scatter on PCA space
- **Hierarchical Clustering** — Ward linkage dendrogram + Agglomerative clustering

### 5. Evaluation
- Accuracy & Error Rate
- Precision / Recall / F1 Score
- Confusion Matrices (all 4 models)
- Side-by-side model comparison bar chart

---

## Results Overview

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | ~98% | ~98% | ~99% | ~98% |
| MLP Classifier | ~97% | ~97% | ~98% | ~98% |
| SVM | ~98% | ~98% | ~99% | ~98% |
| Random Forest | ~97% | ~97% | ~98% | ~97% |

> *Exact values may vary slightly per run due to random state in train/test split.*

---

## Key Findings

- Features like `worst radius`, `worst perimeter`, and `mean concave points` are the strongest class separators
- ~7 PCA components explain 95% of total variance
- K-Means (k=2) clusters align well with true Malignant/Benign labels
- All supervised models achieve >95% accuracy on the test set

---

## Getting Started

### Run in Google Colab (Recommended)
1. Click the **Open in Colab** badge above, or go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `Breast_Cancer_Wisconsin_ML.ipynb`
3. Click **Runtime → Run All**
4. No setup or data download required 

### Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/your-username/breast-cancer-ml-pipeline.git
cd breast-cancer-ml-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook Breast_Cancer_Wisconsin_ML.ipynb
```

---

## Requirements

```txt
scikit-learn>=1.0
matplotlib>=3.5
seaborn>=0.11
pandas>=1.3
numpy>=1.21
scipy>=1.7
jupyter>=1.0
```

> All packages are pre-installed in Google Colab — no manual setup needed.

---

## Repository Structure

```
 breast-cancer-ml-pipeline
 ┣ Breast_Cancer_Wisconsin_ML.ipynb ← Main notebook
 ┣ README.md ← This file
 ┗ requirements.txt ← Python dependencies
```

---

## License

This project is licensed under the **MIT License**.
The Breast Cancer Wisconsin dataset is publicly available via scikit-learn, originally sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

---

## Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — Original dataset source
- [scikit-learn](https://scikit-learn.org/) — ML library and dataset loader
- [Kaggle — Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) — Community reference
