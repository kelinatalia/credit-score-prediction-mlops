# Credit Score Prediction

## Overview
This project predicts a customer's credit score category (Good, Standard, or Poor) based on their demographic data, income, credit history, and payment behavior. It includes a machine learning pipeline, a local Streamlit dashboard, and a cloud deployment on AWS SageMaker with its own Streamlit dashboard.

## Steps
- Exploratory data analysis: checked missing values, duplicates, class distribution, and outliers
- Data preprocessing: cleaned numeric placeholder values, imputed missing data, scaled numeric features, and encoded categorical features (ordinal and one-hot)
- Modeling: compared several algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, XGBoost) with MLflow experiment tracking, and selected the best pipeline
- Deployment: packaged the model as a SageMaker inference endpoint, and built a Streamlit dashboard for user interaction

## App Versions
- `app_streamlit.py`, a local dashboard that loads the trained model directly
- `app_streamlit_cloud.py`, a dashboard that calls the AWS SageMaker endpoint (`inferencing_cloud.py`) instead of loading the model directly, showing a more production style setup

## Tech Stack
Python, pandas, scikit-learn, XGBoost, MLflow, Streamlit, AWS SageMaker, boto3
