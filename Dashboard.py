import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import reload
import seaborn as sns

plt = reload(plt)


def dashboard():
    st.title("Simple Data Dashboard for Data Visualization")
    data_file = st.file_uploader(
        "Select the csv or xlsx file in the working directory", type=["csv", "xlsx"]
    )
    if data_file is not None:
        st.write("Data file uploaded...")
        df = pd.read_excel(data_file)

        st.subheader("Data Preview")
        st.write("First few rows")
        st.write(df.head())

        st.write("Last few rows")
        st.write(df.tail())

        st.write("Data Summary")
        st.write(df.describe())

        st.subheader("Histogram Plot of Data Distribution")

        columns = df.columns.to_list()

        x_column = st.selectbox("Select the x-value you want to plot", columns)
        y_column = st.selectbox("Select the y-value you want to plot", columns)

        if st.checkbox("Generate Plot from your x and y values "):
            plt.figure(figsize=(10, 6))
            plt.plot(df[x_column], df[y_column], marker="s")
            plt.title(f"{y_column} vs {x_column}")
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid(True)
            st.pyplot(plt)

        if st.checkbox("Show Data Distribution"):
            st.write(df.columns.to_list())
            columns_to_plot = df[
                [
                    "ID (mm)",
                    "L (mm)",
                    "Tsat [oC]",
                    "Psat [kPa]",
                    "G [kg/m2s]",
                    "q [W/m2]",
                    "x [-]",
                    "HTC [W/m2 K]",
                    "P_reduced [-]",
                ]
            ]
            fig1, axes = plt.subplots(nrows=3, ncols=3, figsize=(21, 16))
            axes = axes.flatten()

            for i, column in enumerate(columns_to_plot.columns):
                axes[i].hist(df[column], bins=50)
                axes[i].set_xlabel(column)
                axes[i].set_ylabel("Counts")
                axes[i].set_title("")

            fig1.tight_layout()
            st.pyplot(fig1)

        if st.checkbox("Show Data Distribution by Refrigerant"):
            st.write("### Distribution based on Refrigerant")
            fig2, ax = plt.subplots(figsize=(10, 10))
            df["Fluid "].value_counts().plot.pie(
                autopct="%1.1f%%", startangle=90, counterclock=False, ax=ax
            )
            st.pyplot(fig2)

        if st.checkbox("Show Data Distribution by Authors"):
            st.write("### Distribution based on Authors")
            fig3, ax = plt.subplots(figsize=(8, 4))
            order = df["Author"].value_counts().index
            sns.countplot(
                y="Author", data=df, order=order, orient="h", color="#2b8cbe", ax=ax
            )
            ax.set_xlabel("Counts")
            ax.set_ylabel("Authors")
            st.pyplot(fig3)

        if st.checkbox("Correlation Matrix"):
            numerical_columns_ = df.select_dtypes(include=[np.number]).columns.tolist()
            if "HTC [W/m2 K]" in numerical_columns_:
                numerical_columns_.remove("HTC [W/m2 K]")

            numerical_columns = numerical_columns_ + ["HTC [W/m2 K]"]

            corr_Matrix = pd.DataFrame(
                df[numerical_columns], columns=numerical_columns
            ).corr()

            mask = np.triu(np.ones_like(corr_Matrix, dtype=bool))

            fig4, axes4 = plt.subplots(figsize=(35, 30))
            sns.heatmap(
                corr_Matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                vmax=1,
                vmin=-1,
                mask=mask,
                linewidths=0.5,
                cbar_kws={"shrink": 0.75},
            )
            plt.tight_layout()
            # sorted_corr = corr_Matrix["HTC [W/m2 K]"].sort_values(ascending=False)

            st.pyplot(fig4)

        if st.checkbox("Feature Importance"):
            numerical_columns_ = df.select_dtypes(include=[np.number]).columns.tolist()
            if "HTC [W/m2 K]" in numerical_columns_:
                numerical_columns_.remove("HTC [W/m2 K]")

            numerical_columns = numerical_columns_ + ["HTC [W/m2 K]"]

            corr_Matrix = pd.DataFrame(
                df[numerical_columns], columns=numerical_columns
            ).corr()

            sorted_corr = corr_Matrix["HTC [W/m2 K]"].sort_values(ascending=False)
            fig5, axes5 = plt.subplots(figsize=(20, 12))
            sns.barplot(x=sorted_corr.values, y=sorted_corr.index, palette="coolwarm")
            plt.tight_layout()
            plt.grid(True)
            st.pyplot(fig5)

    else:
        st.write("File was not properly uploaded")
