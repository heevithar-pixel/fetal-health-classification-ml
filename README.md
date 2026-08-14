# Fetal Health Classification Using Machine Learning

## a. Problem Statement

Cardiotocography (CTG) records fetal heart-rate and uterine-contraction measurements. The objective of this project is to build and compare multiple machine-learning classification models that classify fetal state from 21 numerical CTG features.

The prediction target is **NSP**, with three classes:

- **1 — Normal**
- **2 — Suspect**
- **3 — Pathologic**

The five classifiers explicitly listed in the assignment are implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

An interactive Streamlit application allows test-data upload, model selection, evaluation and row-level prediction.

> **Educational-use note:** This application is an academic machine-learning demonstration and must not be used for clinical diagnosis or medical decision-making.

---

## b. Dataset Description

**Dataset:** Cardiotocography  
**Source:** UCI Machine Learning Repository  
**Dataset page:** https://archive.ics.uci.edu/dataset/193/cardiotocography  
**DOI:** https://doi.org/10.24432/C51S4N  
**Creators:** D. Campos and J. Bernardes  
**License:** CC BY 4.0

The original dataset contains **2,126 records** and **21 numerical input features**. The CTGs were processed to derive fetal-heart-rate and uterine-contraction measurements and were assigned fetal-state labels by expert obstetricians.

### Data used in this project

Before splitting, the selected 21 features plus `NSP` were validated and **12 exact duplicate records were removed**. Duplicate removal was performed before train/test splitting to prevent identical observations from appearing on both sides of the split.

| Item | Value |
|---|---:|
| Original records | 2,126 |
| Exact duplicate records removed | 12 |
| Clean records used | 2,114 |
| Input features | 21 |
| Target | NSP |
| Classification type | Multiclass (3 classes) |
| Missing/non-numeric values in selected data | 0 |
| Training rows | 1,691 |
| Untouched test rows | 423 |
| Final train-test split | 80:20, stratified |
| Random state | 42 |

### Class distribution after duplicate removal

| NSP | Fetal state | Count | Percentage |
|---:|---|---:|---:|
| 1 | Normal | 1,647 | 77.91% |
| 2 | Suspect | 292 | 13.81% |
| 3 | Pathologic | 175 | 8.28% |

The classes are imbalanced. Therefore, Accuracy is interpreted together with AUC, weighted Precision/Recall/F1, MCC, macro metrics and the class-wise classification report.

### Input features

| Feature | Meaning |
|---|---|
| LB | Fetal heart rate baseline |
| AC | Accelerations per second |
| FM | Fetal movements per second |
| UC | Uterine contractions per second |
| DL | Light decelerations per second |
| DS | Severe decelerations per second |
| DP | Prolonged decelerations per second |
| ASTV | Percentage of time with abnormal short-term variability |
| MSTV | Mean short-term variability |
| ALTV | Percentage of time with abnormal long-term variability |
| MLTV | Mean long-term variability |
| Width | Width of fetal heart rate histogram |
| Min | Minimum of fetal heart rate histogram |
| Max | Maximum of fetal heart rate histogram |
| Nmax | Number of histogram peaks |
| Nzeros | Number of histogram zeros |
| Mode | Histogram mode |
| Mean | Histogram mean |
| Median | Histogram median |
| Variance | Histogram variance |
| Tendency | Histogram tendency |

The original workbook also contains **CLASS**, another expert-derived classification label. `CLASS` is deliberately excluded from the input features because using one target label to predict another could introduce target leakage. Only the 21 measured CTG features are used to predict `NSP`.

---

## c. GitHub Repository Link

**GitHub Repository:**
https://github.com/heevithar-pixel/fetal-health-classification-ml

**Live Streamlit App:**
https://fetal-health-classification-ml-4doq6p7fpo4cm8bd89trus.streamlit.app

The repository contains the complete source code, pinned model-runtime dependencies, README, test CSV, model definitions, saved trained models and generated experiment artifacts.

---

## d. Models Used and Evaluation Metrics

### Experimental methodology

The project uses a separate model-development stage and final evaluation stage:

1. Remove exact duplicate records **before** any split.
2. Create a stratified **80% training / 20% test** split.
3. Keep the 20% test set untouched during model selection.
4. On the training partition only, use **5-fold Stratified Cross-Validation**.
5. Select hyperparameters using **macro F1** because it gives equal importance to each class and is more informative under class imbalance.
6. Refit the selected configuration on the complete training partition.
7. Evaluate each final model once on the untouched test set.

Logistic Regression and KNN use `StandardScaler` inside a scikit-learn `Pipeline`, so scaling is fitted independently inside each CV fold and no preprocessing information leaks from validation/test data.

### Hyperparameter selection

| Model | CV-selected configuration |
|---|---|
| Logistic Regression | `C=10.0`, `class_weight=None` |
| Decision Tree | `max_depth=10`, `min_samples_leaf=2` |
| K-Nearest Neighbor | `n_neighbors=3`, `weights='distance'` |
| Naive Bayes | `var_smoothing=1e-11` |
| Random Forest | `class_weight='balanced'`, `max_depth=None`, `min_samples_leaf=2`, 200 trees |

Class weighting was **evaluated rather than automatically forced**. Cross-validation selected no class weighting for Logistic Regression but selected balanced class weighting for Random Forest.

### 5-fold training CV summary

Values below are **mean ± standard deviation across the five training folds**. They are used for model development, not as a replacement for the final untouched-test results.

| Model | CV Accuracy | CV AUC | CV Precision | CV Recall | CV F1 | CV MCC | CV Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8918 ± 0.0052 | 0.9705 ± 0.0042 | 0.8920 ± 0.0083 | 0.8918 ± 0.0052 | 0.8909 ± 0.0065 | 0.7041 ± 0.0163 | 0.7920 ± 0.0161 |
| Decision Tree | 0.9255 ± 0.0118 | 0.8991 ± 0.0128 | 0.9251 ± 0.0121 | 0.9255 ± 0.0118 | 0.9249 ± 0.0116 | 0.7937 ± 0.0317 | 0.8730 ± 0.0204 |
| K-Nearest Neighbor | 0.9030 ± 0.0180 | 0.9355 ± 0.0255 | 0.9006 ± 0.0199 | 0.9030 ± 0.0180 | 0.9010 ± 0.0194 | 0.7279 ± 0.0537 | 0.8178 ± 0.0348 |
| Naive Bayes | 0.8190 ± 0.0223 | 0.9350 ± 0.0096 | 0.8767 ± 0.0141 | 0.8190 ± 0.0223 | 0.8353 ± 0.0194 | 0.6174 ± 0.0419 | 0.7069 ± 0.0309 |
| Random Forest | 0.9397 ± 0.0100 | 0.9833 ± 0.0056 | 0.9389 ± 0.0100 | 0.9397 ± 0.0100 | 0.9388 ± 0.0100 | 0.8331 ± 0.0274 | 0.8886 ± 0.0160 |

### Final Model Comparison Table — untouched test data

The table below is the main assignment evaluation table. AUC uses multiclass **One-vs-Rest (OvR), weighted averaging**. Precision, Recall and F1 use **weighted averaging** for the required comparison table. MCC uses the multiclass Matthews Correlation Coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9007 | 0.9726 | 0.8947 | 0.9007 | 0.8966 | 0.7166 |
| Decision Tree | 0.9433 | 0.9139 | 0.9418 | 0.9433 | 0.9410 | 0.8397 |
| K-Nearest Neighbor | 0.9173 | 0.9373 | 0.9126 | 0.9173 | 0.9122 | 0.7607 |
| Naive Bayes | 0.8180 | 0.9344 | 0.8771 | 0.8180 | 0.8358 | 0.6067 |
| **Random Forest (Ensemble)** | **0.9527** | **0.9876** | **0.9515** | **0.9527** | **0.9511** | **0.8672** |

### Additional imbalance-aware test metrics

Weighted averages are retained in the mandatory assignment table because they account for class support. Macro and class-wise metrics are additionally examined so that performance on the smaller Suspect and Pathologic classes is not hidden by the dominant Normal class.

| Model | Macro Precision | Macro Recall | Macro F1 | Macro AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8330 | 0.7881 | 0.8083 | 0.9709 |
| Decision Tree | 0.9284 | 0.8744 | 0.8983 | 0.9166 |
| K-Nearest Neighbor | 0.8769 | 0.7838 | 0.8239 | 0.9249 |
| Naive Bayes | 0.6745 | 0.7471 | 0.6907 | 0.8940 |
| **Random Forest** | **0.9425** | **0.8965** | **0.9174** | **0.9877** |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline with 0.9007 test Accuracy and 0.9726 weighted AUC. Cross-validation selected `class_weight=None`, showing that forced balancing did not improve the model-selection objective. Its macro F1 (0.8083) is lower than its weighted F1 because performance is less even across the minority classes. |
| Decision Tree | The tuned tree (`max_depth=10`, `min_samples_leaf=2`) achieves 0.9433 Accuracy and 0.8397 MCC. Constraining the tree prevents the near-perfect training memorization seen with an unrestricted tree while retaining strong holdout performance. Its AUC is lower than the Random Forest because a single tree gives less stable probability estimates. |
| K-Nearest Neighbor | CV selected `k=3` with distance weighting after scaling. It reaches 0.9173 Accuracy and 0.7607 MCC, but its macro Recall (0.7838) indicates weaker sensitivity across the smaller classes than the two tree-based models. |
| Naive Bayes | Has the lowest Accuracy (0.8180), macro F1 (0.6907) and MCC (0.6067). Several histogram variables are strongly correlated — for example Mean/Median ≈ 0.95 and Median/Mode ≈ 0.93 — which weakens Gaussian Naive Bayes' conditional-independence assumption and can contribute to poorer discrimination. |
| Random Forest (Ensemble) | Gives the best overall holdout performance: 0.9527 Accuracy, 0.9876 weighted AUC, 0.9511 weighted F1, 0.9174 macro F1 and 0.8672 MCC. CV selected balanced class weighting and `min_samples_leaf=2`, providing strong overall and minority-aware performance. |
| **Overall Winner** | **Random Forest** is selected as the overall winner because it achieves the highest Accuracy, AUC, weighted F1, macro F1 and MCC on the untouched test set. |

### Random Forest model insight

The project also exports Random Forest feature importance as an additional interpretation aid. The highest-ranked model features include **ASTV, ALTV, Mean, AC, Median and DP**. These importances describe how the fitted model uses the variables; they **do not imply clinical causation**.

---

## Streamlit Application

**Live Streamlit App:** `PASTE_STREAMLIT_APP_LINK_HERE`

The application includes:

- CSV test-data upload
- model-selection dropdown
- comparison of all five classifiers on labeled test data
- Accuracy, AUC, Precision, Recall, F1 and MCC
- additional macro metrics for imbalance-aware interpretation
- confusion matrix
- classification report
- row-level fetal-state predictions
- prediction confidence
- downloadable prediction CSV
- Random Forest feature-importance visualization
- CV model-development summary
- bundled `test_data.csv` so evaluation results are visible immediately
- explicit model-loading errors/warnings instead of silently retraining a different model
- cached parsing/evaluation so switching models does not unnecessarily recompute all results

---

## Repository Structure

```text
ML_Assignment_2_Fetal_Health_Improved/
│
├── app.py
├── config.py
├── ml_utils.py
├── train_models.py
├── requirements.txt
├── README.md
├── RUN_GUIDE.md
├── SUBMISSION_CONTENT.md
├── test_data.csv
│
├── data/
│   ├── cardiotocography.csv
│   ├── cardiotocography_clean.csv
│   └── train_data.csv
│
├── model/
│   ├── __init__.py
│   ├── model_factory.py
│   ├── logistic_regression_model.py
│   ├── decision_tree_model.py
│   ├── knn_model.py
│   ├── naive_bayes_model.py
│   ├── random_forest_model.py
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   └── model_metadata.json
│
└── artifacts/
    ├── class_distribution.csv
    ├── model_metrics.csv
    ├── model_metrics_extended.csv
    ├── cv_model_selection.csv
    ├── random_forest_feature_importance.csv
    └── random_forest_feature_importance.png
```

---

## Reproducibility and Deployment

The saved models were generated with:

- **scikit-learn 1.8.0**
- **joblib 1.5.3**

These are pinned in `requirements.txt` to avoid saved-model incompatibility. The other numerical packages are also pinned to the versions used to generate the supplied artifacts. This environment requires **Python 3.11 or newer**. For Streamlit Community Cloud, select a Python 3.11+ runtime.

### Run locally / on BITS Virtual Lab

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

For the required BITS Virtual Lab evidence, capture one screenshot after `python train_models.py` finishes and shows all five model runs plus `Training completed successfully.`

---

## References

1. Campos, D. & Bernardes, J. (2000). *Cardiotocography* [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C51S4N.
2. UCI Machine Learning Repository — Cardiotocography: https://archive.ics.uci.edu/dataset/193/cardiotocography
