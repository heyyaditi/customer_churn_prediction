# main.py - runs the complete customer churn project
# run with:  python main.py

import pickle

from src import data_loader, preprocessing, analysis, visualization, utils
import config


def main():
    utils.create_folders()

    # 1. load and understand data
    utils.print_heading("STEP 1: LOAD DATA")
    df = data_loader.load_raw_data()
    data_loader.understand_data(df)

    # 2. cleaning
    utils.print_heading("STEP 2: DATA CLEANING")
    df, report = preprocessing.clean_data(df)
    print("Rows before cleaning :", report["rows_before"])
    print("Duplicate rows       :", report["duplicates"])
    print("Missing TotalCharges :", report["missing_total_charges"])
    print("Missing SupportCalls :", report["missing_support_calls"])
    print("Rows after cleaning  :", report["rows_after"])
    print("Missing after        :", report["missing_after"])
    data_loader.save_processed_data(df)

    # 3. EDA
    utils.print_heading("STEP 3: EDA")
    print("Churn percentage:", analysis.churn_percentage(df), "%")
    for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]:
        print("\nChurn rate by", col)
        print(analysis.churn_rate_by(df, col))

    df = analysis.add_tenure_groups(df)
    print("\nChurn rate by tenure group")
    print(analysis.churn_rate_by(df, "TenureGroup"))

    # 4. charts
    utils.print_heading("STEP 4: CHARTS")
    visualization.plot_churn_distribution(df)
    visualization.plot_churn_by_contract(df)
    visualization.plot_churn_by_tenure(df)
    visualization.plot_monthly_charges_box(df)
    visualization.plot_churn_by_payment(df)
    visualization.plot_correlation_heatmap(df)
    visualization.plot_churn_by_services(df)
    print("Charts saved in", config.CHART_DIR)

    # 5. model
    utils.print_heading("STEP 5: MODEL")
    X, y = preprocessing.encode_features(df)
    print("Features used:", list(X.columns))
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = preprocessing.split_and_scale(X, y)
    print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])

    model = analysis.train_model(X_train_s, y_train)
    results = analysis.evaluate_model(model, X_test_s, y_test)

    print("Accuracy :", round(results["accuracy"], 4))
    print("Precision:", round(results["precision"], 4))
    print("Recall   :", round(results["recall"], 4))
    print("F1 score :", round(results["f1"], 4))
    print("Confusion matrix:")
    print(results["confusion_matrix"])
    print(results["report"])

    visualization.plot_confusion_matrix(results["confusion_matrix"])
    coef = analysis.get_coefficients(model, X.columns)
    visualization.plot_feature_importance(coef)

    # 6. high risk customers
    utils.print_heading("STEP 6: HIGH RISK CUSTOMERS")
    high_risk = analysis.find_high_risk(df, model, scaler, list(X.columns))
    high_risk.to_csv(config.HIGH_RISK_PATH, index=False)
    print("High risk customers (probability >=", config.HIGH_RISK_LIMIT, "):", len(high_risk))

    # 7. save model and report
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "columns": list(X.columns)}, f)

    text = "Customer Churn Prediction - Model Report\n"
    text += "Model: Logistic Regression\n"
    text += "Accuracy : " + str(round(results["accuracy"], 4)) + "\n"
    text += "Precision: " + str(round(results["precision"], 4)) + "\n"
    text += "Recall   : " + str(round(results["recall"], 4)) + "\n"
    text += "F1 score : " + str(round(results["f1"], 4)) + "\n\n"
    text += "Confusion matrix:\n" + str(results["confusion_matrix"]) + "\n\n"
    text += results["report"]
    utils.save_text(text, config.REPORT_PATH)
    print("Done. Check the outputs folder.")


if __name__ == "__main__":
    main()
