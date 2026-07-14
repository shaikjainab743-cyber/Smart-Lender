import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pandas.api.types import is_numeric_dtype

# Load dataset
df = pd.read_csv("dataset/loan_prediction.csv")

# Fill missing values
for col in df.columns:
    if is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
encoder = LabelEncoder()

for col in df.columns:
    if not is_numeric_dtype(df[col]):
        df[col] = encoder.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/loan_model.pkl")

print("Model trained successfully!")
