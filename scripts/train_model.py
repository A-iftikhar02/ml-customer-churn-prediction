# scripts/train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# --- load & clean ---
df = pd.read_csv(CSV_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df = df.drop(columns=["customerID"])

# target to 0/1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# features/target
y = df["Churn"]
X = df.drop(columns=["Churn"])

# dtypes
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
# treat all object cols as categorical
categorical_features = X.select_dtypes(include="object").columns.tolist()

# --- preprocessing ---
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop"
)

# --- pipeline ---
clf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=2, random_state=42
        ))
    ]
)

# split, train, save
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
clf.fit(X_train, y_train)
joblib.dump({"pipeline": clf, "feature_order": X.columns.tolist()}, MODEL_PATH)

print(f"✅ Saved pipeline to: {MODEL_PATH}")
print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")
