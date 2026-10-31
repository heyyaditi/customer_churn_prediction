# visualization.py - all the charts (saved as png in outputs/charts)

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config

sns.set_style("whitegrid")
COLORS = ["#1F4E78", "#E07B39"]


def save_chart(name):
    path = os.path.join(config.CHART_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_churn_distribution(df):
    counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts()
    plt.figure(figsize=(5, 4))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=COLORS, startangle=90)
    plt.title("Churn Distribution")
    save_chart("01_churn_distribution.png")


def plot_churn_by_contract(df):
    rate = df.groupby("Contract")["Churn"].mean() * 100
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=rate.index, y=rate.values, color=COLORS[0])
    ax.bar_label(ax.containers[0], fmt="%.1f%%")
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn Rate (%)")
    save_chart("02_churn_by_contract.png")


def plot_churn_by_tenure(df):
    rate = df.groupby("TenureGroup", observed=True)["Churn"].mean() * 100
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=rate.index.astype(str), y=rate.values, color=COLORS[0])
    ax.bar_label(ax.containers[0], fmt="%.1f%%")
    plt.title("Churn Rate by Tenure (months)")
    plt.xlabel("Tenure Group")
    plt.ylabel("Churn Rate (%)")
    save_chart("03_churn_by_tenure.png")


def plot_monthly_charges_box(df):
    data = df.copy()
    data["Status"] = data["Churn"].map({0: "Stayed", 1: "Churned"})
    plt.figure(figsize=(5, 4))
    sns.boxplot(x="Status", y="MonthlyCharges", data=data, palette=COLORS, hue="Status", legend=False)
    plt.title("Monthly Charges vs Churn")
    save_chart("04_monthly_charges_vs_churn.png")


def plot_churn_by_payment(df):
    rate = (df.groupby("PaymentMethod")["Churn"].mean() * 100).sort_values()
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x=rate.values, y=rate.index, color=COLORS[0])
    ax.bar_label(ax.containers[0], fmt="%.1f%%")
    plt.title("Churn Rate by Payment Method")
    plt.xlabel("Churn Rate (%)")
    plt.ylabel("")
    save_chart("05_churn_by_payment_method.png")


def plot_correlation_heatmap(df):
    cols = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "SupportCalls", "Churn"]
    plt.figure(figsize=(7, 5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="Blues")
    plt.title("Correlation Heatmap")
    save_chart("06_correlation_heatmap.png")


def plot_churn_by_services(df):
    # TechSupport and OnlineSecurity side by side
    plt.figure(figsize=(7, 4))
    for i, col in enumerate(["TechSupport", "OnlineSecurity"]):
        plt.subplot(1, 2, i + 1)
        rate = df.groupby(col)["Churn"].mean() * 100
        ax = sns.barplot(x=rate.index, y=rate.values, color=COLORS[0])
        ax.bar_label(ax.containers[0], fmt="%.0f%%")
        plt.title(col)
        plt.xlabel("")
        plt.xticks(rotation=20, fontsize=8)
        plt.ylabel("Churn Rate (%)" if i == 0 else "")
    save_chart("07_churn_by_services.png")


def plot_confusion_matrix(cm):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    save_chart("08_confusion_matrix.png")


def plot_feature_importance(coef):
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:10].sort_values()
    colors = [COLORS[1] if v > 0 else COLORS[0] for v in top.values]
    plt.figure(figsize=(7, 5))
    plt.barh(top.index, top.values, color=colors)
    plt.title("Top 10 Model Coefficients (orange = more churn)")
    plt.xlabel("Coefficient")
    save_chart("09_model_coefficients.png")
