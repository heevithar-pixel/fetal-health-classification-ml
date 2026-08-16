# Fetal Health Classification Using Machine Learning

## Problem Statement

Cardiotocography (CTG) records fetal heart-rate and uterine-contraction measurements. The objective of this project is to build and compare multiple machine-learning classification models that classify fetal state from 21 numerical CTG features.

**Prediction target:** `NSP` (3-class multiclass classification)

- `1` — Normal
- `2` — Suspect
- `3` — Pathologic

The following classifiers are implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

An interactive Streamlit application allows CTG test-data upload, model selection, evaluation, and row-level prediction.

---

## Dataset Description

**Dataset:** Cardiotocography  
**Source:** UCI Machine Learning Repository  
**DOI:** 10.24432/C51S4N  
**Creators:** D. Campos and J. Bernardes

The original dataset contains **2,126 records** and **21 numerical input features**. The CTGs were processed to derive fetal-heart-rate and uterine-contraction measurements and were assigned fetal-state labels by expert obstetricians.

### Data Used in This Project

| Item | Value |
|---|---:|
| Original records | 2,126 |
| Exact duplicate records removed | 12 |
| Clean records used | 2,114 |
| Input features | 21 |
| Target | NSP |
| Classification type | Multiclass (3 classes) |
| Missing / non-numeric values in selected data | 0 |
| Training rows | 1,691 |
| Untouched test rows | 423 |
| Final split | 80:20, stratified |
| Random state | 42 |

Duplicate removal was performed before train/test splitting to prevent identical observations from appearing on both sides of the split.

### Class Distribution After Duplicate Removal

| NSP | Fetal State | Count | Percentage |
|---:|---|---:|---:|
| 1 | Normal | 1,647 | 77.91% |
| 2 | Suspect | 292 | 13.81% |
| 3 | Pathologic | 175 | 8.28% |

The classes are imbalanced. Therefore, Accuracy is interpreted together with AUC, weighted Precision/Recall/F1, MCC, macro metrics, and the class-wise classification report.

### Input Features

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

The original workbook also contains `CLASS`, another expert-derived classification label. `CLASS` is deliberately excluded from the input features because using one target label to predict another could introduce target leakage. Only the 21 measured CTG features are used to predict `NSP`.

---

## GitHub Repository

https://github.com/heevithar-pixel/fetal-health-classification-ml

The repository contains the complete source code, runtime dependencies, README, test CSV, model definitions, saved trained models, and generated experiment artifacts.

---

## Experimental Methodology

1. Remove exact duplicate records before any split.
2. Create a stratified 80% training / 20% test split.
3. Keep the 20% test set untouched during model selection.
4. Use 5-fold Stratified Cross-Validation on the training partition only.
5. Select hyperparameters using macro F1 because it gives equal importance to each class under class imbalance.
6. Refit the selected configuration on the complete training partition.
7. Evaluate each final model once on the untouched test set.

Logistic Regression and KNN use `StandardScaler` inside a scikit-learn `Pipeline` so scaling is fitted independently inside each CV fold, preventing preprocessing leakage.

### CV-Selected Configurations

| Model | Selected Configuration |
|---|---|
| Logistic Regression | `C=10.0`, `class_weight=None` |
| Decision Tree | `max_depth=10`, `min_samples_leaf=2` |
| K-Nearest Neighbor | `n_neighbors=3`, `weights='distance'` |
| Naive Bayes | `var_smoothing=1e-11` |
| Random Forest | `class_weight='balanced'`, `max_depth=None`, `min_samples_leaf=2`, `n_estimators=200` |

---

## Final Model Comparison — Untouched Test Data

AUC uses multiclass One-vs-Rest (OvR) with weighted averaging. Precision, Recall and F1 use weighted averaging for the required comparison table. MCC uses the multiclass Matthews Correlation Coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9007 | 0.9726 | 0.8947 | 0.9007 | 0.8966 | 0.7166 |
| Decision Tree | 0.9433 | 0.9139 | 0.9418 | 0.9433 | 0.9410 | 0.8397 |
| K-Nearest Neighbor | 0.9173 | 0.9373 | 0.9126 | 0.9173 | 0.9122 | 0.7607 |
| Naive Bayes | 0.8180 | 0.9344 | 0.8771 | 0.8180 | 0.8358 | 0.6067 |
| **Random Forest (Ensemble)** | **0.9527** | **0.9876** | **0.9515** | **0.9527** | **0.9511** | **0.8672** |

### Additional Imbalance-Aware Metrics

| Model | Macro Precision | Macro Recall | Macro F1 | Macro AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8330 | 0.7881 | 0.8083 | 0.9709 |
| Decision Tree | 0.9284 | 0.8744 | 0.8983 | 0.9166 |
| K-Nearest Neighbor | 0.8769 | 0.7838 | 0.8239 | 0.9249 |
| Naive Bayes | 0.6745 | 0.7471 | 0.6907 | 0.8940 |
| **Random Forest** | **0.9425** | **0.8965** | **0.9174** | **0.9877** |

---

## Observations on Model Performance

| ML Model | Observation |
|---|---|
| Logistic Regression | Provides a strong linear baseline with 0.9007 test Accuracy and 0.9726 weighted AUC. Its macro F1 is lower than its weighted F1, indicating less even performance across minority classes. |
| Decision Tree | The tuned tree achieves 0.9433 Accuracy and 0.8397 MCC. Constraining tree complexity reduces overfitting while retaining strong holdout performance. |
| K-Nearest Neighbor | Cross-validation selected `k=3` with distance weighting after scaling. It reaches 0.9173 Accuracy and 0.7607 MCC, but macro Recall is weaker than the two tree-based models. |
| Naive Bayes | Has the lowest Accuracy, macro F1 and MCC. Several CTG histogram variables are strongly correlated, which weakens Gaussian Naive Bayes' conditional-independence assumption. |
| Random Forest (Ensemble) | Gives the best overall holdout performance with 0.9527 Accuracy, 0.9876 weighted AUC, 0.9511 weighted F1, 0.9174 macro F1 and 0.8672 MCC. |
| **Overall Winner** | **Random Forest** is selected as the overall winner because it achieves the highest Accuracy, AUC, weighted F1, macro F1 and MCC on the untouched test set. |

---

## Random Forest Model Insight

The Random Forest model also provides feature importance scores to show which variables contributed most to its predictions. Features such as **ASTV, ALTV, Mean, AC, Median, and DP** were among the most influential. These scores indicate the relative importance of the features within the model and should not be interpreted as evidence of clinical cause-and-effect.

---

## Streamlit Application

**Live App:**  
https://fetal-health-classification-ml-4doq6p7fpg4cm8bd89trus.streamlit.app/

The final interface is titled **CTG Fetal State Classifier** and includes:

- CSV CTG test-data upload with input-schema validation
- classifier-selection dropdown
- comparison of all five classifiers on labeled test data
- Accuracy, AUC, Precision, Recall, F1, and MCC
- macro-averaged diagnostics for imbalance-aware interpretation
- confusion matrix and classification report
- row-level fetal-state predictions and prediction confidence
- downloadable prediction CSV
- Random Forest feature-importance visualization
- 5-fold Stratified CV model-selection summary
- bundled `test_data.csv` for immediate evaluation
- explicit validation errors when an uploaded CSV does not contain the required 21 CTG features

---

## Repository Structure

```text
fetal-health-classification-ml/
|-- app.py
|-- train_models.py
|-- config.py
|-- ml_utils.py
|-- requirements.txt
|-- README.md
|-- RUN_GUIDE.md
|-- SUBMISSION_CONTENT.md
|-- test_data.csv
|-- data/
|-- model/
`-- artifacts/
```

---

## Reproducibility and Deployment

The saved models were generated with **scikit-learn 1.8.0** and **joblib 1.5.3**. These versions are pinned in `requirements.txt` to reduce saved-model compatibility issues. The supplied environment requires Python 3.11 or newer.

### Run Locally / on BITS Virtual Lab

```bash
python3 -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

---

## References

1. Campos, D. & Bernardes, J. (2000). *Cardiotocography* [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C51S4N
2. UCI Machine Learning Repository — Cardiotocography: https://archive.ics.uci.edu/dataset/193/cardiotocography

