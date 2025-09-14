import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Config
# -----------------------------
DATA_CSV = "flood_cleaned_for_submission.csv"
MODEL_FILE = "model.joblib"
TARGET_COLUMN = "FloodProbability"
THRESHOLD = 0.5  # probability threshold for classification label


# -----------------------------
# Train or load model
# -----------------------------
def train_and_save_model(csv_path: str, model_path: str):
    st.write("🔄 Training model...")

    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        st.error(f"Target column '{TARGET_COLUMN}' not found in CSV!")
        st.stop()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Only keep numeric columns
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    feature_names = X_numeric.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Save
    joblib.dump({"pipeline": pipeline, "feature_names": feature_names}, model_path)
    st.success(f"✅ Model trained and saved as {model_path}")

    return pipeline, feature_names


def load_model(model_path: str):
    obj = joblib.load(model_path)
    return obj["pipeline"], obj["feature_names"]


# -----------------------------
# Load or train model
# -----------------------------
if Path(MODEL_FILE).exists():
    pipeline, FEATURE_NAMES = load_model(MODEL_FILE)
else:
    if not Path(DATA_CSV).exists():
        st.error(
            f"No model found and CSV not present. Place '{DATA_CSV}' in folder or provide model.joblib"
        )
        st.stop()
    pipeline, FEATURE_NAMES = train_and_save_model(DATA_CSV, MODEL_FILE)


# -----------------------------
# Prediction function
# -----------------------------
def predict_from_list(values):
    arr = np.array(values, dtype=float).reshape(1, -1)
    prob = pipeline.predict(arr)[0]
    prob_clamped = float(max(0.0, min(1.0, prob)))
    label = "🌊 Flood Predicted" if prob_clamped >= THRESHOLD else "✅ No Flood Predicted"
    return prob_clamped, label


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌊 Flood Probability Predictor")
st.markdown("Enter feature values below to predict flood probability.")

# Input fields
inputs = {}
for f in FEATURE_NAMES:
    inputs[f] = st.number_input(f"Enter {f}", value=0.0)

# Predict button
if st.button("Predict"):
    values = [inputs[f] for f in FEATURE_NAMES]
    prob, label = predict_from_list(values)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Probability:** {prob:.3f}")
    st.write(f"**Label:** {label}")

st.markdown("---")
st.caption("Built with Streamlit, scikit-learn, and joblib 🚀")
