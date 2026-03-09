import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1. Load Dataset
# =========================

file_path = r"C:\Users\Jawad\Downloads\OHLC_without_shift_4h.csv"

df = pd.read_csv(file_path)

# Keep only required columns
df = df[["open", "high", "low", "close", "volume", "prediction"]]

# =========================
# 2. Create Sensitive Feature
# =========================

# This will be used only for fairness analysis
df["trader_group"] = np.random.choice([0, 1], size=len(df))

# =========================
# 3. Create Target
# =========================

df["target"] = (df["prediction"] > df["close"]).astype(int)

print("Target distribution:")
print(df["target"].value_counts())

# =========================
# 4. Define Features
# =========================

# Full dataset including sensitive feature
X_full = df[["open", "high", "low", "close", "volume", "trader_group"]]

y = df["target"]

# =========================
# 5. Train Test Split
# =========================

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42
)

# Remove sensitive column for training
X_train = X_train_full.drop(columns=["trader_group"])
X_test = X_test_full.drop(columns=["trader_group"])

# =========================
# 6. Train Model
# =========================

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# 7. Evaluate Model
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# =========================
# 8. Save Model Files
# =========================

# Model
joblib.dump(model, "model.pkl")

# Feature names
feature_names = list(X_train.columns)
joblib.dump(feature_names, "features.pkl")

# Test dataset for fairness evaluation
joblib.dump(X_test_full, "X_test_full.pkl")

# Test labels
joblib.dump(y_test, "y_test.pkl")

print("\nSaved files:")
print("model.pkl")
print("features.pkl")
print("X_test_full.pkl")
print("y_test.pkl")