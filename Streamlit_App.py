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
    #df = pd.read_csv(uploaded_file, sep=";")
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.write("Columns:", df.columns)

    target_column = st.selectbox("Select Target Column", df.columns)

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    

    model_path = f"models/{model_option.lower().replace(' ', '_')}.pkl"
    model = joblib.load(model_path)

    

    preds = model.predict(X)

    st.subheader("📈 Classification Report")
    st.text(classification_report(y, preds))

    st.subheader("🧮 Confusion Matrix")
    st.write(confusion_matrix(y, preds))
