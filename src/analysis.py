# analysis.py - EDA numbers, model training and evaluation

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import config


def churn_percentage(df):
    return round(df["Churn"].mean() * 100, 2)


def churn_rate_by(df, column):
    # churn percentage for each category of a column
    result = df.groupby(column)["Churn"].mean() * 100
    return result.round(2).sort_values(ascending=False)


def add_tenure_groups(df):
    df = df.copy()
    df["TenureGroup"] = pd.cut(df["Tenure"], bins=[0, 12, 24, 48, 72],
                               labels=["0-12", "13-24", "25-48", "49-72"])
    return df


def train_model(X_train_scaled, y_train):
    # class_weight balanced is used because churners are only about 25% of data
    model = LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=config.RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]),
    }
    return results


def get_coefficients(model, columns):
    # positive coefficient = more chance of churn
    coef = pd.Series(model.coef_[0], index=columns)
    return coef.sort_values(ascending=False)


def find_high_risk(df, model, scaler, columns):
    # predict churn probability for every customer
    from src.preprocessing import encode_features
    X, y = encode_features(df)
    X = X.reindex(columns=columns, fill_value=0)
    prob = model.predict_proba(scaler.transform(X))[:, 1]

    result = df[["CustomerID", "Tenure", "Contract", "MonthlyCharges", "PaymentMethod"]].copy()
    result["ChurnProbability"] = prob.round(3)
    result = result[result["ChurnProbability"] >= config.HIGH_RISK_LIMIT]
    return result.sort_values("ChurnProbability", ascending=False)
