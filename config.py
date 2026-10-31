# config.py - paths and constants used in the project

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "customer_churn_clean.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
MODEL_PATH = os.path.join(OUTPUT_DIR, "churn_model.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "model_report.txt")
HIGH_RISK_PATH = os.path.join(OUTPUT_DIR, "high_risk_customers.csv")

TARGET = "Churn"

# features suggested in the project problem statement
NUMERIC_FEATURES = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# customers with churn probability above this are called high risk
HIGH_RISK_LIMIT = 0.80
