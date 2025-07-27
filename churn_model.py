# churn_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# ------------------------------
# 1. Load and prepare the data
# ------------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")

# Clean and preprocess
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df)

# ------------------------------
# 2. Feature selection
# ------------------------------
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# ------------------------------
# 3. Train/Test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ------------------------------
# 4. Train logistic regression model
# ------------------------------
model = LogisticRegression(class_weight='balanced', max_iter=2000)
model.fit(X_train, y_train)

# ------------------------------
# 5. Predictions and evaluation
# ------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:", round(roc_auc_score(y_test, y_proba), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------
# 6. Export the trained model
# ------------------------------
import joblib
joblib.dump(model, "model/logistic_churn_model.pkl")
print("\nModel saved as logistic_churn_model.pkl")
