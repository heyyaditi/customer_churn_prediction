# preprocessing.py - data cleaning, encoding and train/test split

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config


def clean_data(df):
    df = df.copy()
    report = {}
    report["rows_before"] = len(df)
    report["duplicates"] = int(df.duplicated().sum())

    # TotalCharges has blank spaces, so it was read as text
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    report["missing_total_charges"] = int(df["TotalCharges"].isnull().sum())
    report["missing_support_calls"] = int(df["SupportCalls"].isnull().sum())

    # remove duplicate rows
    df.drop_duplicates(inplace=True)

    # fill missing values with median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["SupportCalls"] = df["SupportCalls"].fillna(df["SupportCalls"].median())
    df["SupportCalls"] = df["SupportCalls"].astype(int)

    # fix text columns (extra spaces and small/capital letters)
    for col in ["Gender", "Contract", "InternetService", "PaymentMethod",
                "TechSupport", "OnlineSecurity", "StreamingTV", "Churn"]:
        df[col] = df[col].str.strip()
    df["Gender"] = df["Gender"].str.title()

    # target column Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = df.reset_index(drop=True)
    report["rows_after"] = len(df)
    report["missing_after"] = int(df.isnull().sum().sum())
    return df, report


def encode_features(df):
    # select the features and convert categories to numbers
    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    X = pd.get_dummies(X, columns=config.CATEGORICAL_FEATURES, drop_first=True, dtype=int)
    y = df[config.TARGET]
    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
