"""
Streamlit UI for the Customer Churn classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "churn-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.title("Customer Churn Predictor")
st.write(
    "Enter the customer's profile below to predict whether they will "
    "churn via SageMaker."
)

# 1. Setup User Inputs
# Column order MUST match FEATURE_NAMES in inference.py
# (customer_churn.csv with CustomerID dropped).
age = st.slider("Age", 18, 70, 30)
gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
tenure = st.slider("Tenure (months)", 0, 60, 12)
usage_frequency = st.slider("Usage Frequency", 0, 30, 10)
support_calls = st.slider("Support Calls", 0, 10, 2)
payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)
subscription_type = st.selectbox(
    "Subscription Type", ["Basic", "Standard", "Premium"]
)
contract_length = st.selectbox(
    "Contract Length", ["Monthly", "Quarterly", "Annual"]
)
total_spend = st.slider("Total Spend", 100, 1000, 500)
last_interaction = st.slider("Last Interaction (days)", 0, 30, 10)

if st.button("Predict", type="primary"):
    features = [
        age,
        gender,
        tenure,
        usage_frequency,
        support_calls,
        payment_delay,
        subscription_type,
        contract_length,
        total_spend,
        last_interaction,
    ]
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs = result["probabilities"][0]

        if label == "churned":
            st.error(f"Predicted outcome: **{label}**")
        else:
            st.success(f"Predicted outcome: **{label}**")

        st.write("Class probabilities:")
        st.bar_chart(
            {
                "probability": [
                    {"class": "stayed",  "probability": probs[0]},
                    {"class": "churned", "probability": probs[1]},
                ]
            },
            x="class",
            y="probability",
        )
