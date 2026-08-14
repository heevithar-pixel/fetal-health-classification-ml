import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from config import (
    ARTIFACT_DIR,
    CLASS_LABELS,
    CLEAN_DATA_PATH,
    CV_FOLDS,
    DATA_PATH,
    FEATURES,
    MODEL_DIR,
    MODEL_FILES,
    RANDOM_STATE,
    TARGET,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
)
from ml_utils import evaluate_model
from model.model_factory import build_search_specs


def clean_dataset(data: pd.DataFrame):
    """Validate required columns and remove exact duplicate records before splitting."""
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    selected = data[required].copy()
    for column in required:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    if selected.isna().any().any():
        bad = selected.columns[selected.isna().any()].tolist()
        raise ValueError("Missing/non-numeric values found in: " + ", ".join(bad))

    selected[TARGET] = selected[TARGET].astype(int)
    invalid_targets = sorted(set(selected[TARGET].unique()) - set(CLASS_LABELS))
    if invalid_targets:
        raise ValueError(f"Unexpected NSP labels: {invalid_targets}")

    original_rows = len(selected)
    duplicate_rows = int(selected.duplicated().sum())
    selected = selected.drop_duplicates().reset_index(drop=True)
    return selected, original_rows, duplicate_rows


def build_scoring():
    return {
        "accuracy": "accuracy",
        "auc_weighted": "roc_auc_ovr_weighted",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
        "mcc": make_scorer(matthews_corrcoef),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    raw_data = pd.read_csv(DATA_PATH)
    data, original_rows, duplicate_rows = clean_dataset(raw_data)
    data.to_csv(CLEAN_DATA_PATH, index=False)

    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_data = x_train.copy()
    train_data[TARGET] = y_train.values
    train_data.to_csv(TRAIN_PATH, index=False)

    test_data = x_test.copy()
    test_data[TARGET] = y_test.values
    test_data.to_csv(TEST_PATH, index=False)

    class_distribution = (
        data[TARGET]
        .value_counts()
        .sort_index()
        .rename_axis(TARGET)
        .reset_index(name="Count")
    )
    class_distribution["Fetal State"] = class_distribution[TARGET].map(CLASS_LABELS)
    class_distribution["Percentage"] = (
        100 * class_distribution["Count"] / len(data)
    )
    class_distribution.to_csv(ARTIFACT_DIR / "class_distribution.csv", index=False)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = build_scoring()

    assignment_rows = []
    extended_rows = []
    cv_rows = []
    best_params = {}
    trained_models = {}

    print("\nFetal Health Classification - Improved Model Training")
    print("=" * 76)
    print(f"Original rows: {original_rows}")
    print(f"Exact duplicate rows removed before split: {duplicate_rows}")
    print(f"Clean rows: {len(data)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Untouched test rows: {len(test_data)}")
    print(f"CV: {CV_FOLDS}-fold Stratified CV on TRAINING data only")
    print("Selection metric: macro F1")
    print("Target: NSP (1=Normal, 2=Suspect, 3=Pathologic)\n")

    for model_name, (estimator, param_grid) in build_search_specs().items():
        print(f"Tuning {model_name} ...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            refit="f1_macro",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        trained_models[model_name] = model
        joblib.dump(model, MODEL_DIR / MODEL_FILES[model_name])

        metrics, _, _ = evaluate_model(model, x_test, y_test)
        train_metrics, _, _ = evaluate_model(model, x_train, y_train)

        assignment_rows.append(
            {
                "ML Model Name": model_name,
                **{k: metrics[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]},
            }
        )
        extended_rows.append(
            {
                "ML Model Name": model_name,
                **metrics,
                "Training Accuracy": train_metrics["Accuracy"],
                "Training Macro F1": train_metrics["Macro F1"],
            }
        )

        idx = search.best_index_
        cv_record = {
            "ML Model Name": model_name,
            "Best Parameters": json.dumps(search.best_params_, sort_keys=True),
            "CV Macro F1 Mean": search.cv_results_["mean_test_f1_macro"][idx],
            "CV Macro F1 Std": search.cv_results_["std_test_f1_macro"][idx],
            "CV Precision Mean": search.cv_results_["mean_test_precision_weighted"][idx],
            "CV Precision Std": search.cv_results_["std_test_precision_weighted"][idx],
            "CV Recall Mean": search.cv_results_["mean_test_recall_weighted"][idx],
            "CV Recall Std": search.cv_results_["std_test_recall_weighted"][idx],
            "CV Weighted F1 Mean": search.cv_results_["mean_test_f1_weighted"][idx],
            "CV Weighted F1 Std": search.cv_results_["std_test_f1_weighted"][idx],
            "CV Accuracy Mean": search.cv_results_["mean_test_accuracy"][idx],
            "CV Accuracy Std": search.cv_results_["std_test_accuracy"][idx],
            "CV AUC Mean": search.cv_results_["mean_test_auc_weighted"][idx],
            "CV AUC Std": search.cv_results_["std_test_auc_weighted"][idx],
            "CV MCC Mean": search.cv_results_["mean_test_mcc"][idx],
            "CV MCC Std": search.cv_results_["std_test_mcc"][idx],
        }
        cv_rows.append(cv_record)
        best_params[model_name] = search.best_params_

        print(f"  Best params: {search.best_params_}")
        print(
            f"  CV macro F1={cv_record['CV Macro F1 Mean']:.4f} ± {cv_record['CV Macro F1 Std']:.4f} | "
            f"Test Accuracy={metrics['Accuracy']:.4f} | Test Macro F1={metrics['Macro F1']:.4f} | MCC={metrics['MCC']:.4f}"
        )

    assignment_df = pd.DataFrame(assignment_rows)
    extended_df = pd.DataFrame(extended_rows)
    cv_df = pd.DataFrame(cv_rows)
    assignment_df.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)
    extended_df.to_csv(ARTIFACT_DIR / "model_metrics_extended.csv", index=False)
    cv_df.to_csv(ARTIFACT_DIR / "cv_model_selection.csv", index=False)

    rf_model = trained_models["Random Forest"]
    feature_importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    feature_importance.to_csv(ARTIFACT_DIR / "random_forest_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = feature_importance.head(12).sort_values("Importance")
    ax.barh(top["Feature"], top["Importance"])
    ax.set_title("Random Forest - Top 12 Feature Importances")
    ax.set_xlabel("Feature importance")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "random_forest_feature_importance.png", dpi=180)
    plt.close(fig)

    metadata = {
        "dataset_rows_original": original_rows,
        "duplicate_rows_removed": duplicate_rows,
        "dataset_rows_clean": len(data),
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "cv_selection_metric": "macro F1",
        "target": TARGET,
        "features": FEATURES,
        "best_parameters": best_params,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    winner = extended_df.loc[extended_df["Macro F1"].idxmax(), "ML Model Name"]
    print("\n" + "=" * 76)
    print(f"Overall winner by untouched-test macro F1: {winner}")
    print("Assignment metrics: artifacts/model_metrics.csv")
    print("Extended metrics: artifacts/model_metrics_extended.csv")
    print("CV selection summary: artifacts/cv_model_selection.csv")
    print("Feature importance: artifacts/random_forest_feature_importance.csv")
    print("Models saved in model/*.joblib")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
