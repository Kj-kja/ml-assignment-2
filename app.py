import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.title("Breast Cancer Classification")
st.write(
    "Compare five machine learning models on the "
    "Breast Cancer Wisconsin Diagnostic dataset."
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors (KNN)": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine (SVM)": "svm.pkl",
}

model_name = st.selectbox(
    "Select a classification model",
    list(MODEL_FILES.keys()),
)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the test data and target column",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if "target" not in data.columns:
            st.error("The CSV must contain a column named 'target'.")
            st.stop()

        X = data.drop(columns=["target"])
        y = data["target"]

        model_path = BASE_DIR / MODEL_FILES[model_name]
        scaler_path = BASE_DIR / "scaler.pkl"

        if not model_path.exists():
            st.error(f"Model file not found: {model_path.name}")
            st.stop()

        model = joblib.load(model_path)

        # Scale features for models trained with StandardScaler.
        if model_name in [
            "Logistic Regression",
            "K-Nearest Neighbors (KNN)",
            "Support Vector Machine (SVM)",
        ]:
            if not scaler_path.exists():
                st.error("The scaler.pkl file is missing.")
                st.stop()

            scaler = joblib.load(scaler_path)
            X_model = scaler.transform(X)
        else:
            X_model = X

        predictions = model.predict(X_model)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_model)[:, 1]
        else:
            probabilities = None

        st.subheader("Uploaded Data")
        st.write(f"Number of records: {len(data)}")
        st.dataframe(data.head())

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy_score(y, predictions):.4f}")
        col2.metric(
            "Precision",
            f"{precision_score(y, predictions, zero_division=0):.4f}",
        )
        col3.metric(
            "Recall",
            f"{recall_score(y, predictions, zero_division=0):.4f}",
        )

        col4, col5, col6 = st.columns(3)

        col4.metric("F1 Score", f"{f1_score(y, predictions, zero_division=0):.4f}")
        col5.metric("MCC", f"{matthews_corrcoef(y, predictions):.4f}")

        if probabilities is not None and y.nunique() == 2:
            auc = roc_auc_score(y, probabilities)
            col6.metric("AUC", f"{auc:.4f}")
        else:
            col6.metric("AUC", "Unavailable")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, predictions, labels=[0, 1])

        st.write("Rows = actual labels; columns = predicted labels.")
        st.write("Label 0 = malignant; label 1 = benign.")

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual malignant (0)", "Actual benign (1)"],
                columns=["Predicted malignant (0)", "Predicted benign (1)"],
            )
        )

        st.subheader("Classification Report")
        report = classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Malignant", "Benign"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"Could not process this CSV: {e}")
        st.info(
            "Check that the CSV has the same feature columns as the "
            "training dataset and includes a 'target' column."
        )
else:
    st.info("Upload test_data.csv to evaluate the selected model.")
