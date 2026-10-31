# dashboard.py - Streamlit dashboard for the churn project
# run with:  streamlit run dashboard.py

import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import config
from src import data_loader, analysis

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")


@st.cache_data
def get_data():
    df = data_loader.load_processed_data()
    return analysis.add_tenure_groups(df)


def get_model():
    with open(config.MODEL_PATH, "rb") as f:
        return pickle.load(f)


if not os.path.exists(config.PROCESSED_DATA_PATH):
    st.error("Processed data not found. Please run main.py first.")
    st.stop()

df = get_data()

st.title("Customer Churn Prediction Dashboard")
st.caption("Telecom customer data | Python, Pandas, Scikit-learn, Streamlit")

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Contract", df["Contract"].unique(), default=list(df["Contract"].unique()))
internet = st.sidebar.multiselect("Internet Service", df["InternetService"].unique(), default=list(df["InternetService"].unique()))
payments = st.sidebar.multiselect("Payment Method", df["PaymentMethod"].unique(), default=list(df["PaymentMethod"].unique()))
tenure_range = st.sidebar.slider("Tenure (months)", int(df["Tenure"].min()), int(df["Tenure"].max()),
                                 (int(df["Tenure"].min()), int(df["Tenure"].max())))

data = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet) &
          df["PaymentMethod"].isin(payments) &
          df["Tenure"].between(tenure_range[0], tenure_range[1])]

if len(data) == 0:
    st.warning("No customers match the selected filters.")
    st.stop()

# ---------- KPI row ----------
st.subheader("Key Numbers")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers", len(data))
k2.metric("Churned Customers", int(data["Churn"].sum()))
k3.metric("Churn Rate", str(round(data["Churn"].mean() * 100, 2)) + "%")
k4.metric("Avg Monthly Charges", str(round(data["MonthlyCharges"].mean(), 2)))
k5.metric("Avg Tenure", str(round(data["Tenure"].mean(), 1)) + " months")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Churn Analysis", "Model Results", "Predict Customer"])

# ---------- tab 1 ----------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = data["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#1F4E78", "#E07B39"], startangle=90)
        ax.set_title("Churn Distribution")
        st.pyplot(fig)
    with c2:
        rate = data.groupby("Contract")["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=rate.index, y=rate.values, color="#1F4E78", ax=ax)
        ax.set_title("Churn Rate by Contract")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        rate = data.groupby("TenureGroup", observed=True)["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=rate.index.astype(str), y=rate.values, color="#1F4E78", ax=ax)
        ax.set_title("Churn Rate by Tenure Group (months)")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)
    with c4:
        rate = data.groupby("PaymentMethod")["Churn"].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=rate.values, y=rate.index, color="#1F4E78", ax=ax)
        ax.set_title("Churn Rate by Payment Method")
        ax.set_xlabel("Churn Rate (%)")
        st.pyplot(fig)

# ---------- tab 2 ----------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        plot_data = data.copy()
        plot_data["Status"] = plot_data["Churn"].map({0: "Stayed", 1: "Churned"})
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(x="Status", y="MonthlyCharges", data=plot_data, ax=ax)
        ax.set_title("Monthly Charges vs Churn")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        cols = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "SupportCalls", "Churn"]
        sns.heatmap(data[cols].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    st.write("Churn rate (%) by column")
    column = st.selectbox("Select a column", ["InternetService", "TechSupport", "OnlineSecurity",
                                              "StreamingTV", "Gender", "SeniorCitizen", "SupportCalls"])
    st.bar_chart(analysis.churn_rate_by(data, column))

# ---------- tab 3 ----------
with tab3:
    if not os.path.exists(config.REPORT_PATH):
        st.warning("Model report not found. Please run main.py first.")
    else:
        st.write("Logistic Regression results on the test data")
        with open(config.REPORT_PATH) as f:
            st.text(f.read())
        c1, c2 = st.columns(2)
        cm_path = os.path.join(config.CHART_DIR, "08_confusion_matrix.png")
        coef_path = os.path.join(config.CHART_DIR, "09_model_coefficients.png")
        if os.path.exists(cm_path):
            c1.image(cm_path)
        if os.path.exists(coef_path):
            c2.image(coef_path)

    if os.path.exists(config.HIGH_RISK_PATH):
        st.subheader("High Risk Customers")
        high_risk = pd.read_csv(config.HIGH_RISK_PATH)
        st.write("Customers with churn probability of", config.HIGH_RISK_LIMIT, "or more:", len(high_risk))
        st.dataframe(high_risk.head(50))
        st.download_button("Download high risk list", high_risk.to_csv(index=False),
                           file_name="high_risk_customers.csv")

# ---------- tab 4 ----------
with tab4:
    st.write("Enter customer details to predict churn")
    if not os.path.exists(config.MODEL_PATH):
        st.warning("Model not found. Please run main.py first.")
    else:
        c1, c2, c3 = st.columns(3)
        tenure = c1.number_input("Tenure (months)", 1, 72, 12)
        monthly = c1.number_input("Monthly Charges", 18.0, 120.0, 70.0)
        senior = c1.selectbox("Senior Citizen", [0, 1])
        contract = c2.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
        payment = c2.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
        internet_service = c2.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
        tech = c3.selectbox("Tech Support", ["Yes", "No", "No Internet Service"])
        security = c3.selectbox("Online Security", ["Yes", "No", "No Internet Service"])

        if st.button("Predict"):
            saved = get_model()
            customer = pd.DataFrame([{
                "Tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
                "SeniorCitizen": senior, "Contract": contract, "PaymentMethod": payment,
                "InternetService": internet_service, "TechSupport": tech, "OnlineSecurity": security}])
            customer = pd.get_dummies(customer, columns=config.CATEGORICAL_FEATURES, dtype=int)
            customer = customer.reindex(columns=saved["columns"], fill_value=0)
            prob = saved["model"].predict_proba(saved["scaler"].transform(customer))[0][1]

            st.metric("Churn Probability", str(round(prob * 100, 1)) + "%")
            if prob >= config.HIGH_RISK_LIMIT:
                st.error("High risk customer - retention offer recommended")
            elif prob >= 0.5:
                st.warning("Medium risk customer")
            else:
                st.success("Low risk customer")
