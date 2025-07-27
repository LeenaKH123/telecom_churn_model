# inspect_model.py

import joblib

# Load the model
model = joblib.load("model/logistic_churn_model.pkl")

# Inspect basic properties
print("\nModel type:", type(model))
print("\nIntercept:", model.intercept_)
print("\nNumber of coefficients:", len(model.coef_[0]))
print("\nSample of coefficients:", model.coef_[0][:10])  # Print first 10
