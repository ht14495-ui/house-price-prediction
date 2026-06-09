import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv("housing.csv")

TARGET_COL = "median_house_value"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ── 2. Train/test split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Preprocessing pipeline ────────────────────────────
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# ── 4. Build & train best model ───────────────────────────
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", HistGradientBoostingRegressor(
        l2_regularization=0.1,
        learning_rate=0.1,
        max_depth=None,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        random_state=42
    ))
])

model.fit(X_train, y_train)
print("Model trained successfully!")

# ── 5. Save model to disk ─────────────────────────────────
joblib.dump(model, "model.pkl")
print("model.pkl saved!")