import streamlit as st
import json
from pathlib import Path


REPORT_PATH = Path("monitoring/drift_report.json")


st.set_page_config(
    page_title="Telco Churn MLOps Monitoring",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Telecom Churn MLOps Monitoring Dashboard")
st.write("Real-time data drift monitoring using the Kolmogorov-Smirnov test.")


if not REPORT_PATH.exists():

    st.error("Drift report not found.")

    st.info(
        "Run: python monitoring\\drift_monitor.py"
    )

else:

    with open(REPORT_PATH, "r") as file:
        report = json.load(file)

    overall_status = report["overall_status"]

    st.subheader("Overall Drift Status")

    if overall_status == "DRIFT DETECTED":
        st.error("🔴 DRIFT DETECTED")
        st.warning(
            "Data distribution has changed in one or more features."
        )
    else:
        st.success("🟢 NO SIGNIFICANT DRIFT")
        st.info(
            "No significant data drift was detected."
        )

    st.divider()

    st.subheader("Feature Drift Analysis")

    features = report["features"]

    for feature in features:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Feature",
                feature["feature"]
            )

        with col2:
            st.metric(
                "KS Statistic",
                feature["ks_statistic"]
            )

        with col3:
            st.metric(
                "P-value",
                feature["p_value"]
            )

        with col4:

            if feature["status"] == "DRIFT DETECTED":
                st.error("🔴 Drift Detected")
            else:
                st.success("🟢 No Drift")

        st.divider()

    st.caption(
        "Drift threshold: p-value < 0.05"
    )