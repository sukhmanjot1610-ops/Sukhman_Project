import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="❤️ Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
)

# -------------------- CUSTOM STYLE -------------------------
st.markdown("""
<style>
body {background-color: #f7f9fc;}
.big-font {font-size: 32px !important; font-weight: 700;}
.subtitle {font-size: 20px !important; color:#ff4b4b; font-weight: 600;}
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #ddd;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER IMAGE -------------------------
st.image("heartpic.jpg",width=350, caption="A Healthy Heart is a Happy Life ❤️")

st.markdown("<p class='big-font' style='text-align:center;'>Heart Disease Analysis & Machine Learning Prediction Dashboard</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle' style='text-align:center;'>Explore data, visualize patterns, and see how ML models perform!</p>", unsafe_allow_html=True)
st.write("---")

# -------------------- LOAD DATA -------------------------
df = pd.read_csv("heart.csv")

# Sidebar menu
menu = st.sidebar.selectbox("📌 Navigation Menu", ["🏠 Home", "📊 Data Explorer", "📈 Model Evaluation"])

# -------------------- HOME SECTION -------------------------
if menu == "🏠 Home":
    st.subheader("✨ Welcome to the Heart Disease Prediction Dashboard")
    st.write("""
    This dashboard allows you to **discover trends**, **visualize medical attributes**, 
    and **compare machine learning models** for predicting heart disease.
    
    💡 *Health is wealth — let's analyze it scientifically!*    
    """)
    st.dataframe(df.head())

    st.success("Dataset Loaded Successfully ✔")

# -------------------- DATA EXPLORER -------------------------
elif menu == "📊 Data Explorer":
    st.header("🔍 Data Visualization & Exploration")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Heart Disease Distribution")
        fig = plt.figure()
        sns.countplot(x="HeartDisease", data=df, palette="Set2")
        st.pyplot(fig)

    with col2:
        st.write("### Age Distribution")
        fig = plt.figure()
        sns.histplot(df["Age"], kde=True, color="royalblue")
        st.pyplot(fig)

    st.write("### Chest Pain Type vs Heart Disease")
    fig = plt.figure()
    sns.countplot(x="ChestPainType", hue="HeartDisease", data=df, palette="coolwarm")
    plt.xticks(rotation=0)
    st.pyplot(fig)

# -------------------- MODEL SECTION -------------------------
elif menu == "📈 Model Evaluation":
    st.header("🤖 Machine Learning Model Comparison")

    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

    X = df_encoded.drop("HeartDisease", axis=1)
    y = df_encoded["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = [
        LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42),
    ]

    results = []
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": model.__class__.__name__,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)

    st.write("### 📋 Model Performance Results")
    st.dataframe(results_df, use_container_width=True)

    st.write("### 📈 Performance Comparison Graph")
    fig = plt.figure(figsize=(10,6))
    results_df.set_index("Model")[["Accuracy","Precision","Recall","F1 Score"]].plot(kind="bar", colormap="viridis")
    plt.ylim(0, 1.1)
    plt.grid(axis="y")
    st.pyplot(fig)

    st.success("Evaluation Complete 🎉")

# -------------------- FOOTER -------------------------
st.write("---")
st.markdown("Made with ❤️ by Sukhman | Stay Healthy, Stay Happy 🌸")
