# Telecom Churn Prediction Project

This project aims to explore and predict customer churn in a telecom company using real-world data and machine learning.

## 🔍 Part 1: Exploratory Data Analysis (EDA)
- Clean and preprocess the customer dataset
- Identify key churn drivers (e.g. contract type, tenure, monthly charges)
- Visualize behavioral patterns with charts

## 🤖 Part 2: Predictive Modeling
- Train a logistic regression model to predict churn
- Evaluate performance using ROC-AUC, precision, recall
- Export the trained model for future scoring

## 📁 Project Structure
├── eda_churn.py              # Part 1: EDA script
├── churn_model.py            # Part 2: Model training and evaluation
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
├── model/
│   └── logistic_churn_model.pkl
├── images/
│   ├── churn_by_tenure.png
│   ├── churn_by_contract.png
│   └── churn_by_charges.png
📂 Output
Model saved as model/logistic_churn_model.pkl

Charts saved in images/

💡 Author
Leena K — Medium Profile (https://medium.com/@leenamk)


