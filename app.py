import streamlit as st
import pandas as pd
from core.engine import SPDSEngine

st.set_page_config(page_title="SPDS", page_icon="📊", layout="wide")
st.title("SPDS — Strategic & Professional Data Science")
st.caption("Autonomous data analysis and decision-support workspace")

uploaded = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Upload a dataset to begin.")
    st.markdown("""
### Included
- Automatic profiling and data-quality checks
- Exploratory summaries and correlations
- IQR-based anomaly screening
- Baseline Random Forest prediction
- Automated business findings and next-step suggestions

This is a decision-support prototype; important decisions should be validated with current data and domain expertise.
""")
    st.stop()

try:
    df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

engine = SPDSEngine(df)
p = engine.profile()
st.success(f"Loaded {p['rows']:,} rows × {p['columns']:,} columns")

a, b, c, d = st.columns(4)
a.metric("Rows", f"{p['rows']:,}")
b.metric("Columns", f"{p['columns']:,}")
c.metric("Missing cells", f"{p['missing_cells']:,}")
d.metric("Duplicate rows", f"{p['duplicate_rows']:,}")

t1, t2, t3, t4 = st.tabs(["Profile", "Quality & EDA", "Prediction", "Business Intelligence"])

with t1:
    st.dataframe(engine.column_profile(), use_container_width=True)

with t2:
    st.subheader("Quality")
    st.dataframe(engine.quality_report(), use_container_width=True)
    st.subheader("Numeric summary")
    ns = engine.numeric_summary()
    st.dataframe(ns, use_container_width=True) if not ns.empty else st.info("No numeric columns.")
    st.subheader("Correlations")
    cr = engine.correlation_matrix()
    st.dataframe(cr, use_container_width=True) if not cr.empty else st.info("Not enough numeric columns.")
    st.subheader("Potential anomalies")
    ar = engine.anomaly_report()
    st.dataframe(ar, use_container_width=True) if not ar.empty else st.info("No anomaly candidates detected.")

with t3:
    target = st.selectbox("Target column", ["— none —"] + list(df.columns))
    if target != "— none —":
        r = engine.predict(target)
        if r["ok"]:
            st.metric("Model", r["model"])
            st.metric("Validation score", r["score_text"])
            st.write(r["explanation"])
            st.dataframe(r["feature_importance"], use_container_width=True)
        else:
            st.warning(r["message"])

with t4:
    st.subheader("Automated findings")
    for x in engine.business_findings():
        st.markdown(f"- {x}")
    st.subheader("Recommended next steps")
    for x in engine.next_steps():
        st.markdown(f"- {x}")
