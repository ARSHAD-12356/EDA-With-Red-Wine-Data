import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Streamlit page settings
st.set_page_config(page_title="EDA App", layout="wide")
st.title("📊 EDA Dashboard with Red Wine Data")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data Preview
    st.subheader("🔍 Data Preview")
    st.write(df.head())

    # Shape Info
    st.subheader("📏 Dataset Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # Null Values
    st.subheader("📊 Null Values in Dataset")
    st.write(df.isnull().sum())

    # Descriptive Stats
    st.subheader("📈 Descriptive Statistics")
    st.write(df.describe())

    # Correlation Heatmap
    st.subheader("📌 Correlation Heatmap")
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig_corr)

    # Histogram
    st.subheader("📉 Histogram of Any Column")
    selected_col = st.selectbox("Select column for histogram", df.columns)
    fig_hist, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax)
    st.pyplot(fig_hist)

    # Scatter Plot
    st.subheader("🔁 Scatter Plot (X vs Y)")
    x_col = st.selectbox("Select X-axis", df.columns)
    y_col = st.selectbox("Select Y-axis", df.columns)
    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}", size_max=10)
    st.plotly_chart(fig)

    # Categorical Box Plot
    st.subheader("📦 Categorical Box Plot (Seaborn Catplot)")
    cat_x = st.selectbox("Select Categorical Column", df.columns, key='catx')
    num_y = st.selectbox("Select Numerical Column", df.columns, key='numy')
    fig_cat = sns.catplot(x=cat_x, y=num_y, data=df, kind="box", height=5, aspect=2)
    st.pyplot(fig_cat)

    # Matplotlib Bar Plot (value_counts)
    st.subheader("📊 Bar Plot (Matplotlib Value Counts)")
    bar_col = st.selectbox("Select column for Bar Plot", df.columns, key='bar2')
    fig_bar, ax = plt.subplots()
    df[bar_col].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_xlabel(bar_col)
    ax.set_ylabel("Count")
    st.pyplot(fig_bar)

else:
    st.info("👈 Please upload a CSV file to get started.")
