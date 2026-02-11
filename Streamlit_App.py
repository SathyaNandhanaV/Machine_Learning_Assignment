import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Bank ML Classifier", layout="wide")

st.title("📊 Bank Marketing Classification App")

model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.file_uploader("Upload Test CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")

    model_path = f"models/{model_option.lower().replace(' ', '_')}.pkl"
    model = joblib.load(model_path)

    X = df.drop("income", axis=1)
    y = df["income"]

    preds = model.predict(X)

    st.subheader("📈 Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("🧮 Confusion Matrix")
    st.write(confusion_matrix(y, preds))
