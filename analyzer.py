import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from autogluon.tabular import TabularPredictor

st.set_page_config(page_title="AutoGluon Analyzer", layout="wide")
st.title("🤖 Highest Accuracy Analyzer with AutoGluon")

# --- Step 1: Upload Data ---
st.sidebar.header("1. Upload Your Dataset")
data_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if data_file:
    file_type = data_file.name.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(data_file)
    else:
        df = pd.read_excel(data_file)

    st.write(f"✅ Loaded dataset with shape {df.shape}")
    st.dataframe(df.head())

    # --- Step 2: Select Target Column ---
    st.sidebar.header("2. Select Target Column")
    target = st.sidebar.selectbox("Choose the column to predict", df.columns)

    if target:
        # AutoGluon expects a file path for training
        st.subheader("⚙️ Training with AutoGluon")
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            df.to_csv(train_path, index=False)

            with st.spinner("Training the best model with AutoGluon..."):
                predictor = TabularPredictor(label=target, path=tmpdir).fit(train_path)

            st.success("🎯 Training Complete!")
            leaderboard = predictor.leaderboard(silent=True)
            st.write("### 🏆 Leaderboard of Models")
            st.dataframe(leaderboard)

            # --- Step 3: View Predictions ---
            st.subheader("📈 Predictions on the Training Data")
            preds = predictor.predict(df.drop(columns=[target]))
            st.dataframe(pd.DataFrame({
                "True": df[target].values,
                "Predicted": preds
            }).head())

            acc = (df[target] == preds).mean()
            st.write(f"✅ Accuracy on training data: `{acc:.4f}`")

            # --- Step 4: Predict on New Data ---
            st.subheader("🔮 Make Predictions on New Data")
            new_file = st.file_uploader("Upload New Data (CSV or Excel)", key="predict")
            if new_file:
                new_type = new_file.name.split('.')[-1]
                new_df = pd.read_csv(new_file) if new_type == 'csv' else pd.read_excel(new_file)

                if target in new_df.columns:
                    new_df = new_df.drop(columns=[target])

                new_preds = predictor.predict(new_df)
                st.write("### ✨ Predictions")
                new_df['Prediction'] = new_preds
                st.dataframe(new_df)

                csv = new_df.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

else:
    st.info("👈 Upload a dataset to get started.")

