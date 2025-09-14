# retrain_model.py
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

# CSV file containing your training data
DATA_CSV = "flood_cleaned_for_submission.csv"
MODEL_FILE = "model.joblib"
TARGET_COLUMN = "FloodProbability"

print("Checking CSV...")
if not Path(DATA_CSV).exists():
    raise SystemExit(f"CSV not found: {DATA_CSV}")

df = pd.read_csv(DATA_CSV)
if TARGET_COLUMN not in df.columns:
    raise SystemExit(f"Target column '{TARGET_COLUMN}' not found. Columns: {df.columns.tolist()}")

X = df.drop(columns=[TARGET_COLUMN])
X_numeric = X.select_dtypes(include=[np.number]).copy()
feature_names = X_numeric.columns.tolist()

if len(feature_names) == 0:
    raise SystemExit("No numeric feature columns found in CSV.")

X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, df[TARGET_COLUMN], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
])

print("Training model...")
pipeline.fit(X_train, y_train)

joblib.dump({"pipeline": pipeline, "feature_names": feature_names}, MODEL_FILE)
print(f"✅ Model saved to {MODEL_FILE}")
