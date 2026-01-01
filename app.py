import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# =============================
# APP CONFIG
# =============================
st.set_page_config(
    page_title="Campus Cafeteria Satisfaction Analysis",
    layout="wide"
)

st.title("🍽 Campus Cafeteria Satisfaction Analysis")
st.caption("Insight-driven Data Mining using EDA, Preprocessing & Visual Analytics")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("university_cafeteria_survey_dataset1.csv")

df = load_data()

ratings = [
    'Food_Taste', 'Hygiene', 'Pricing',
    'Waiting_Time', 'Meal_Variety', 'Staff_Behavior'
]

# =============================
# SIDEBAR
# =============================
st.sidebar.header("📌 Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Dataset Overview", "EDA & Preprocessing", "Visual Analysis", "Insights & Recommendations"]
)

# =============================
# DATASET OVERVIEW
# =============================
if section == "Dataset Overview":
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ℹ Dataset Info")
        st.write(df.info())
    with col2:
        st.subheader("❗ Missing Values")
        st.write(df.isnull().sum())

# =============================
# EDA + PREPROCESSING (SAME PAGE)
# =============================
elif section == "EDA & Preprocessing":
    st.subheader("🔍 Exploratory Data Analysis (EDA)")

    with st.expander("📊 Statistical Summary"):
        st.
