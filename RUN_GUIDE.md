# Run Guide — Fetal Health Classification

## 0. Check Python version

The pinned model environment uses scikit-learn 1.8.0 and requires **Python 3.11+**.

```bash
python --version
```

For Streamlit Community Cloud, select a Python 3.11+ runtime.

## 1. Open the project folder

```bash
cd ML_Assignment_2_Fetal_Health_Improved
```

## 2. Install the pinned dependencies

```bash
pip install -r requirements.txt
```

## 3. Train, tune and evaluate the models

```bash
python train_models.py
```

The script will:

1. validate the 21 features and NSP target;
2. remove exact duplicate records before any split;
3. create a stratified 80/20 train/test split;
4. use 5-fold Stratified Cross-Validation only on the training set;
5. tune model settings using macro F1 as the selection metric;
6. fit the selected model on all training rows;
7. evaluate once on the untouched test set;
8. save the final models, metrics, CV summary and feature importance.

For the BITS Virtual Lab evidence, take one screenshot after this command finishes. Ideally show the terminal with the five model names, the CV/test results and `Training completed successfully.`

## 4. Run the Streamlit app

```bash
streamlit run app.py
```

Confirm that:

- the default `test_data.csv` loads;
- the model dropdown changes the detailed evaluation;
- all six assignment metrics are visible;
- the confusion matrix and classification report appear;
- CSV upload works.

## 5. GitHub

Create a repository and push the project through normal incremental commits. Replace the placeholder GitHub link in `README.md` and `SUBMISSION_CONTENT.md`.

## 6. Streamlit Community Cloud

Deploy `app.py` from the GitHub repository. Verify the public URL in a signed-out/incognito browser, then replace the Streamlit placeholder link.

## 7. Final PDF

Create one PDF in the assignment's required order:

1. GitHub repository link
2. Live Streamlit application link
3. BITS Virtual Lab execution screenshot
4. README content
