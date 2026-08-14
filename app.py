import io
import pickle
import warnings
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from config import (
    ARTIFACT_DIR,
    BASE_DIR,
    CLASS_LABELS,
    FEATURES,
    MODEL_DIR,
    MODEL_FILES,
    TARGET,
    TEST_PATH,
)
from ml_utils import evaluate_model, normalize_target

st.set_page_config(
    page_title="Fetal Health Classification",
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.25rem; font-weight: 750; margin-bottom: .2rem;}
    .subtle {color: #6b7280; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models():
    """Load the pre-trained models and surface any compatibility warning."""
    loaded = {}
    compatibility_warnings = []

    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required saved model is missing: {path.name}. Run train_models.py first."
            )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", InconsistentVersionWarning)
                loaded[name] = joblib.load(path)
                compatibility_warnings.extend(str(item.message) for item in caught)
        except (pickle.UnpicklingError, EOFError, ImportError, ModuleNotFoundError, AttributeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not load {path.name}. This is commonly caused by incompatible "
                "scikit-learn/joblib versions. Reinstall requirements.txt and retrain if needed."
            ) from exc

    return loaded, compatibility_warnings


@st.cache_data(show_spinner=False)
def parse_csv(csv_bytes: bytes):
    return pd.read_csv(io.BytesIO(csv_bytes))


def validate_features(data: pd.DataFrame):
    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    x_eval = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    if x_eval.isna().any().any():
        bad_columns = x_eval.columns[x_eval.isna().any()].tolist()
        raise ValueError(
            "Non-numeric or missing values were found in: " + ", ".join(bad_columns)
        )
    return x_eval


@st.cache_data(show_spinner=False)
def evaluate_labeled_data(csv_bytes: bytes):
    data = parse_csv(csv_bytes)
    x_eval = validate_features(data)
    y_eval = normalize_target(data[TARGET])
    loaded_models, _ = load_models()

    comparison_rows = []
    detailed_results = {}
    for model_name, model in loaded_models.items():
        result, predictions, probabilities = evaluate_model(model, x_eval, y_eval)
        comparison_rows.append(
            {
                "ML Model Name": model_name,
                **{k: result[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]},
            }
        )
        detailed_results[model_name] = {
            "metrics": result,
            "predictions": predictions,
            "probabilities": probabilities,
        }
    return pd.DataFrame(comparison_rows), detailed_results


def display_metric(value):
    return "N/A" if pd.isna(value) else f"{value:.4f}"


try:
    models, model_warnings = load_models()
except (FileNotFoundError, RuntimeError) as exc:
    st.error(str(exc))
    st.info("From the project folder, run:  python train_models.py")
    st.stop()

if model_warnings:
    st.warning(
        "Saved-model compatibility warning detected. Use the pinned versions in requirements.txt."
    )

st.markdown(
    '<div class="main-title">Fetal Health Classification Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtle">Compare five tuned classifiers using 21 cardiotocography measurements.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Project Controls")
    uploaded_file = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="The CSV must contain all 21 CTG feature columns. Include NSP to calculate evaluation metrics.",
    )
    selected_model = st.selectbox("Select model", list(models.keys()))
    st.divider()
    st.subheader("Target classes")
    st.write("**1** — Normal")
    st.write("**2** — Suspect")
    st.write("**3** — Pathologic")
    st.caption("Educational ML demonstration only — not for clinical decision-making.")

if uploaded_file is None:
    csv_bytes = TEST_PATH.read_bytes()
    data_source = "Bundled assignment test_data.csv"
else:
    csv_bytes = uploaded_file.getvalue()
    data_source = uploaded_file.name

try:
    eval_data = parse_csv(csv_bytes)
    x_eval = validate_features(eval_data)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(str(exc))
    st.stop()

has_target = TARGET in eval_data.columns
if has_target:
    try:
        y_eval = normalize_target(eval_data[TARGET])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
else:
    y_eval = None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows loaded", f"{len(eval_data):,}")
c2.metric("Input features", len(FEATURES))
c3.metric("Models available", len(models))
c4.metric("Evaluation labels", "Available" if has_target else "Prediction only")
st.caption(f"Data source: {data_source}")

if has_target:
    comparison_df, detailed_results = evaluate_labeled_data(csv_bytes)

    st.subheader("Model Comparison on Current Test Data")
    st.dataframe(
        comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
    )

    selected = detailed_results[selected_model]
    selected_metrics = selected["metrics"]
    selected_predictions = selected["predictions"]
    selected_probabilities = selected["probabilities"]

    st.subheader(f"Detailed Evaluation — {selected_model}")
    metric_columns = st.columns(6)
    metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for column, metric_name in zip(metric_columns, metric_names):
        column.metric(metric_name, display_metric(selected_metrics[metric_name]))

    st.caption(
        "The assignment table reports weighted Precision/Recall/F1. "
        "Macro and class-wise metrics are also inspected so the dominant Normal class does not hide minority-class performance."
    )

    with st.expander("Additional imbalance-aware metrics"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Macro Precision", display_metric(selected_metrics["Macro Precision"]))
        m2.metric("Macro Recall", display_metric(selected_metrics["Macro Recall"]))
        m3.metric("Macro F1", display_metric(selected_metrics["Macro F1"]))
        m4.metric("Macro AUC", display_metric(selected_metrics["Macro AUC"]))

    tab1, tab2, tab3 = st.tabs(
        ["Confusion Matrix", "Classification Report", "Predictions"]
    )

    with tab1:
        matrix = confusion_matrix(y_eval, selected_predictions, labels=[1, 2, 3])
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[CLASS_LABELS[1], CLASS_LABELS[2], CLASS_LABELS[3]],
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix — {selected_model}")
        st.pyplot(fig, clear_figure=True)

    with tab2:
        report = classification_report(
            y_eval,
            selected_predictions,
            labels=[1, 2, 3],
            target_names=[CLASS_LABELS[1], CLASS_LABELS[2], CLASS_LABELS[3]],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    with tab3:
        prediction_output = eval_data.copy()
        prediction_output["Predicted_NSP"] = selected_predictions
        prediction_output["Predicted_Fetal_State"] = [
            CLASS_LABELS[int(value)] for value in selected_predictions
        ]
        prediction_output["Prediction_Confidence"] = selected_probabilities.max(axis=1)
        st.dataframe(prediction_output.head(50), use_container_width=True)
        st.download_button(
            "Download predictions as CSV",
            data=prediction_output.to_csv(index=False).encode("utf-8"),
            file_name="fetal_health_predictions.csv",
            mime="text/csv",
        )
else:
    st.info(
        "NSP was not included in the uploaded file, so the app is showing predictions only. "
        "Upload labeled test data to calculate the six required evaluation metrics."
    )
    chosen = models[selected_model]
    predictions = chosen.predict(x_eval)
    probabilities = chosen.predict_proba(x_eval)

    prediction_output = eval_data.copy()
    prediction_output["Predicted_NSP"] = predictions
    prediction_output["Predicted_Fetal_State"] = [
        CLASS_LABELS[int(value)] for value in predictions
    ]
    prediction_output["Prediction_Confidence"] = probabilities.max(axis=1)

    st.subheader(f"Predictions — {selected_model}")
    st.dataframe(prediction_output, use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        data=prediction_output.to_csv(index=False).encode("utf-8"),
        file_name="fetal_health_predictions.csv",
        mime="text/csv",
    )

cv_path = ARTIFACT_DIR / "cv_model_selection.csv"
if cv_path.exists():
    with st.expander("Model development: 5-fold Stratified CV selections"):
        cv_summary = pd.read_csv(cv_path)[
            [
                "ML Model Name",
                "Best Parameters",
                "CV Macro F1 Mean",
                "CV Macro F1 Std",
                "CV Accuracy Mean",
                "CV AUC Mean",
                "CV MCC Mean",
            ]
        ].copy()
        st.dataframe(
            cv_summary.style.format(
                {
                    "CV Macro F1 Mean": "{:.4f}",
                    "CV Macro F1 Std": "{:.4f}",
                    "CV Accuracy Mean": "{:.4f}",
                    "CV AUC Mean": "{:.4f}",
                    "CV MCC Mean": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "CV was performed only on the training partition. The bundled test_data.csv remained untouched until final evaluation."
        )

importance_path = ARTIFACT_DIR / "random_forest_feature_importance.csv"
if importance_path.exists():
    with st.expander("Model insight: Random Forest feature importance"):
        importance = pd.read_csv(importance_path).head(12).sort_values("Importance")
        fig, ax = plt.subplots(figsize=(8, 5.4))
        ax.barh(importance["Feature"], importance["Importance"])
        ax.set_xlabel("Feature importance")
        ax.set_title("Top 12 Random Forest Feature Importances")
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Feature importance describes contribution to this fitted model; it does not establish medical causation."
        )

with st.expander("View input data preview and feature list"):
    st.dataframe(eval_data.head(20), use_container_width=True)
    st.write("**21 CTG input features:**", ", ".join(FEATURES))
