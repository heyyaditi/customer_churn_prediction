# Customer Churn Prediction

Data Analytics project (Group B) - Cronus Consultants.

The project analyzes telecom customer data and builds a Logistic Regression model
that predicts whether a customer will churn (leave) or stay. A Streamlit dashboard
is also included to explore the results.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit

## Project Structure
```
project_name/
│
├── main.py                 # Entry point
├── config.py               # Configuration & constants
├── dashboard.py            # Streamlit dashboard
├── data/
│   ├── raw/                # customer_churn.csv
│   └── processed/          # customer_churn_clean.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── analysis.py
│   ├── visualization.py
│   └── utils.py
├── requirements.txt
├── README.md
└── outputs/                # charts, model, report, high risk customers
```

## How to Run
1. Install the libraries
   ```
   pip install -r requirements.txt
   ```
2. Run the complete project (cleaning, EDA, charts, model)
   ```
   python main.py
   ```
3. Start the dashboard
   ```
   streamlit run dashboard.py
   ```

## What main.py does
1. Loads `data/raw/customer_churn.csv`
2. Cleans the data (duplicates, missing values, data types) and saves it in `data/processed/`
3. Performs EDA and saves the charts in `outputs/charts/`
4. Encodes categorical columns, splits the data (80/20) and trains Logistic Regression
5. Evaluates the model (accuracy, precision, recall, confusion matrix)
6. Saves the model report and the list of high risk customers in `outputs/`

## Dataset
10,000 customers (plus 120 duplicate rows in the raw file). Target column is `Churn` (Yes/No).
The dataset is a sample/simulated telecom dataset created for this project.
