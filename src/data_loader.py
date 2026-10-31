# data_loader.py - reading and saving the csv files

import pandas as pd
import config


def load_raw_data():
    df = pd.read_csv(config.RAW_DATA_PATH)
    return df


def load_processed_data():
    df = pd.read_csv(config.PROCESSED_DATA_PATH)
    return df


def save_processed_data(df):
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print("Cleaned data saved to", config.PROCESSED_DATA_PATH)


def understand_data(df):
    # basic data understanding (same commands as in the problem statement)
    print("First 5 rows:")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nInfo:")
    df.info()
    print("\nDescribe:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nChurn counts:")
    print(df["Churn"].value_counts())
