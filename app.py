import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(
    page_title="Campus Cafeteria Satisfaction Analysis",
    layout="wide"
)

st.title("🍽 Campus Cafeteria Satisfaction Analysis")
st.write("Insight-driven Data Mining using EDA, Preprocessing, and Visualizations")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("university_cafeteria_survey_dataset1.csv")

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("📌 Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Dataset Overview", "EDA", "Preprocessing", "Visual Analysis", "Insights"]
)

ratings = [
    'Food_Taste',
    'Hygiene',
    'Pricing',
    'Waiting_Time',
    'Meal_Variety',
    'Staff_Behavior'
]

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
if section == "Dataset Overview":
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("ℹ Dataset Information")
    st.write(df.info())

    st.subheader("❗ Missing Values")
    st.write(df.isnull().sum())

# -----------------------------
# EDA
# -----------------------------
elif section == "EDA":
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())

# -----------------------------
# PREPROCESSING
# -----------------------------
elif section == "Preprocessing":
    st.subheader("🧹 Data Preprocessing")

    # Handle missing numeric values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Handle missing categorical values
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna("Not Specified")

    st.success("✅ Missing values handled successfully!")

    st.subheader("Missing Values After Preprocessing")
    st.write(df.isnull().sum())

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df.head(50))

    st.info(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# -----------------------------
# VISUAL ANALYSIS
# -----------------------------
elif section == "Visual Analysis":
    st.subheader("📈 Overall Satisfaction Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Overall_Satisfaction', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Average Cafeteria Ratings")
    fig, ax = plt.subplots()
    df[ratings].mean().plot(kind='bar', ax=ax)
    ax.set_ylabel("Average Score (1–5)")
    st.pyplot(fig)

    st.subheader("📦 Satisfaction vs Cafeteria Features")
    selected_feature = st.selectbox("Select Feature", ratings)
    fig, ax = plt.subplots()
    sns.boxplot(x=df[selected_feature], y=df['Overall_Satisfaction'], ax=ax)
    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[ratings + ['Overall_Satisfaction']].corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax
    )
    st.pyplot(fig)

# -----------------------------
# INSIGHTS
# -----------------------------
elif section == "Insights":
    st.subheader("🧠 Key Data Mining Outcomes")

    st.markdown("""
    **Findings**
    - Hygiene and food taste strongly affect satisfaction  
    - Long waiting times reduce satisfaction  
    - Pricing concerns exist among students  
    - Satisfaction varies across departments  

    **Recommendations**
    - Improve hygiene standards  
    - Enhance food quality  
    - Reduce waiting time  
    - Review pricing strategy  
    """)
