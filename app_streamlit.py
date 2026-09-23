import argparse
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
from inferencing import InferenceService, TEST_CASES, ALL_FEATURES

st.set_page_config(page_title="Credit Score Predictor", layout="wide")

@st.cache_resource
def load_service():
    try:
        return InferenceService()
    except FileNotFoundError as e:
        return e

service = load_service()

st.sidebar.title("Application Info")
st.sidebar.info(
    "Customer Credit Score prediction application (Good / Standard / Poor) "
    "based on the machine learning model trained in pipeline.py."
)

if isinstance(service, Exception):
    st.sidebar.error(str(service))
    st.error("Model not found. Please run `python pipeline.py --data data_D.csv` "
              "first in this directory to generate `final_model_pipeline.pkl` and "
              "`target_mapping.pkl`.")
    st.stop()

st.title("Credit Score Prediction Dashboard")
st.caption("Fill out the customer data below, or use one of the test cases in the sidebar for a quick demo.")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Test Case")
st.sidebar.caption("Representative profile per class, taken from the median or mode of the original training data.")

preset_choice = st.sidebar.selectbox("Select test case", ["-- manual input --", "Good", "Standard", "Poor"])
if st.sidebar.button("Apply this test case"):
    if preset_choice in TEST_CASES:
        for k, v in TEST_CASES[preset_choice].items():
            st.session_state[k] = v
        st.rerun()

DEFAULTS = TEST_CASES['Standard']


def default_of(field):
    return st.session_state.get(field, DEFAULTS[field])


tab1, tab2 = st.tabs(["Demographics & Income", "Accounts, Cards & Loans"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August"],
                     key="Month", index=["January", "February", "March", "April", "May", "June", "July", "August"].index(default_of("Month")))
        st.number_input("Age", min_value=14, max_value=100, key="Age", value=int(default_of("Age")))
        st.selectbox("Occupation", [
            'Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
            'Journalist', 'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician',
            'Scientist', 'Teacher', 'Writer'
        ], key="Occupation", index=[
            'Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
            'Journalist', 'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician',
            'Scientist', 'Teacher', 'Writer'
        ].index(default_of("Occupation")))
    with col2:
        st.number_input("Annual Income ($)", min_value=0.0, key="Annual_Income",
                         value=float(default_of("Annual_Income")), step=1000.0)
        st.number_input("Monthly Inhand Salary ($)", min_value=0.0, key="Monthly_Inhand_Salary",
                         value=float(default_of("Monthly_Inhand_Salary")), step=100.0)
        st.number_input("Monthly Balance ($)", min_value=0.0, key="Monthly_Balance",
                         value=float(default_of("Monthly_Balance")), step=50.0)

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Number of Bank Accounts", min_value=0, max_value=20, key="Num_Bank_Accounts",
                         value=int(default_of("Num_Bank_Accounts")))
        st.number_input("Number of Credit Cards", min_value=0, max_value=20, key="Num_Credit_Card",
                         value=int(default_of("Num_Credit_Card")))
        st.number_input("Average Interest Rate (%)", min_value=0, max_value=50, key="Interest_Rate",
                         value=int(default_of("Interest_Rate")))
        st.number_input("Total Number of Loans", min_value=0, max_value=20, key="Num_of_Loan",
                         value=int(default_of("Num_of_Loan")))
        st.number_input("Number of Different Loan Types", min_value=0, max_value=10, key="Loan_Type_Count",
                         value=int(default_of("Loan_Type_Count")),
                         help="How many different types of loans owned, such as Auto Loan, Mortgage Loan, etc.")
    with c2:
        st.number_input("Average Delay from Due Date (Days)", min_value=-10, max_value=120, key="Delay_from_due_date",
                         value=int(default_of("Delay_from_due_date")))
        st.number_input("Number of Delayed Payments", min_value=0, max_value=50, key="Num_of_Delayed_Payment",
                         value=int(default_of("Num_of_Delayed_Payment")))
        st.number_input("Changed Credit Limit (%)", min_value=-20.0, max_value=50.0, key="Changed_Credit_Limit",
                         value=float(default_of("Changed_Credit_Limit")))
        st.number_input("Number of Credit Inquiries", min_value=0, max_value=30, key="Num_Credit_Inquiries",
                         value=int(default_of("Num_Credit_Inquiries")))
        st.selectbox("Credit Mix", ["Good", "Standard", "Bad"], key="Credit_Mix",
                     index=["Good", "Standard", "Bad"].index(default_of("Credit_Mix")),
                     help="Only these 3 options are recognized by the model, matching the training categories.")
    with c3:
        st.number_input("Outstanding Debt ($)", min_value=0.0, key="Outstanding_Debt",
                         value=float(default_of("Outstanding_Debt")))
        st.slider("Credit Utilization Ratio (%)", min_value=10.0, max_value=60.0, key="Credit_Utilization_Ratio",
                  value=float(default_of("Credit_Utilization_Ratio")))
        st.number_input("Credit History Duration (Months)", min_value=0, key="Credit_History_Months",
                         value=int(default_of("Credit_History_Months")))
        st.selectbox("Payment of Minimum Amount?", ["Yes", "No"], key="Payment_of_Min_Amount",
                     index=["Yes", "No"].index(default_of("Payment_of_Min_Amount")))
        st.number_input("Total Monthly EMI ($)", min_value=0.0, key="Total_EMI_per_month",
                         value=float(default_of("Total_EMI_per_month")))
        st.number_input("Monthly Invested Amount ($)", min_value=0.0, key="Amount_invested_monthly",
                         value=float(default_of("Amount_invested_monthly")))
        st.selectbox("Payment Behaviour", [
            "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
            "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
            "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"
        ], key="Payment_Behaviour", index=[
            "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
            "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
            "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments"
        ].index(default_of("Payment_Behaviour")))

st.markdown("---")
submit_btn = st.button("Predict Credit Score", type="primary", use_container_width=True)

if submit_btn:
    input_dict = {f: st.session_state[f] for f in ALL_FEATURES}

    with st.spinner("Analyzing..."):
        try:
            result = service.predict_one(input_dict)
            prediction = result['prediction']

            if prediction == "Good":
                st.success(f"### Prediction Result: {prediction} Credit Score")
            elif prediction == "Standard":
                st.info(f"### Prediction Result: {prediction} Credit Score")
            else:
                st.error(f"### Prediction Result: {prediction} Credit Score")

            if 'probabilities' in result:
                st.subheader("Class Probabilities")
                proba_df = pd.DataFrame({
                    'Class': list(result['probabilities'].keys()),
                    'Probability': list(result['probabilities'].values())
                }).sort_values('Probability', ascending=False)
                st.bar_chart(proba_df.set_index('Class'))
                st.dataframe(proba_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during model prediction: {e}")