# eda_churn.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Clean the data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# -----------------------------
# Plot 1: Churn by Tenure
# -----------------------------
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='tenure', bins=30, hue='Churn', multiple="stack")
plt.title("Churn by Tenure")
plt.xlabel("Tenure (months)")
plt.ylabel("Number of Customers")
plt.savefig("images/churn_by_tenure.png")
plt.close()

# -----------------------------
# Plot 2: Churn by Contract Type
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.savefig("images/churn_by_contract.png")
plt.close()

# -----------------------------
# Plot 3: Churn by Monthly Charges
# -----------------------------
plt.figure(figsize=(8, 4))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True)
plt.title("Churn by Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Density")
plt.savefig("images/churn_by_charges.png")
plt.close()

print("EDA plots saved to the images/ folder.")

