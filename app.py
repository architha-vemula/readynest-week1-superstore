import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Data Cleaner & EDA", page_icon="🧹", layout="wide")

st.title("🧹 Automated Data Cleaning & EDA")
st.markdown("Upload any CSV — get a cleaned file + instant exploratory analysis.")

uploaded_file = st.file_uploader("Upload your raw CSV file", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

    # ── Cleaning ──────────────────────────────────────────────────────────
    with st.spinner("Cleaning your data..."):
        df = df_raw.copy()

        # 1. Strip whitespace from column names + make lowercase
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # 2. Strip whitespace from string values
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

        # 3. Remove duplicate rows
        dupes_removed = df.duplicated().sum()
        df = df.drop_duplicates()

        # 4. Try to parse date-like columns
        for col in str_cols:
            if df[col].str.contains(r'\d{4}', na=False).mean() > 0.5:
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="ignore")
                except Exception:
                    pass

        # 5. Infer numeric types where possible
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

        # 6. Handle nulls: fill numeric with median, drop rows where >50% null
        num_cols = df.select_dtypes(include=[np.number]).columns
        null_before = df.isnull().sum().sum()
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df = df.dropna(thresh=int(len(df.columns) * 0.5))
        null_after = df.isnull().sum().sum()
        nulls_fixed = null_before - null_after

    # ── Cleaning Summary ──────────────────────────────────────────────────
    st.subheader("✅ Cleaning Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows before", df_raw.shape[0])
    c2.metric("Rows after", df.shape[0])
    c3.metric("Duplicates removed", int(dupes_removed))
    c4.metric("Null values fixed", int(nulls_fixed))

    st.subheader("🧼 Cleaned Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    # ── EDA ───────────────────────────────────────────────────────────────
    st.subheader("📊 Exploratory Data Analysis")

    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(include="object")

    # --- Descriptive stats
    with st.expander("📈 Descriptive Statistics", expanded=True):
        st.dataframe(num_df.describe().T.round(2), use_container_width=True)

    # --- Missing values heatmap
    if df.isnull().sum().sum() > 0:
        with st.expander("🕳️ Remaining Missing Values"):
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.heatmap(df.isnull(), cbar=False, ax=ax, yticklabels=False, cmap="YlOrRd")
            ax.set_title("Missing value map")
            st.pyplot(fig)
            plt.close()

    # --- Numeric distributions
    if not num_df.empty:
        with st.expander("📉 Numeric Column Distributions", expanded=True):
            cols_to_plot = num_df.columns[:8]
            n = len(cols_to_plot)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
            axes = np.array(axes).flatten() if n > 1 else [axes]
            for i, col in enumerate(cols_to_plot):
                axes[i].hist(num_df[col].dropna(), bins=30, color="#5DCAA5", edgecolor="white")
                axes[i].set_title(col, fontsize=11)
                axes[i].set_xlabel("")
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # --- Correlation heatmap
    if num_df.shape[1] >= 2:
        with st.expander("🔗 Correlation Heatmap"):
            fig, ax = plt.subplots(figsize=(min(10, num_df.shape[1] * 1.2), min(8, num_df.shape[1])))
            corr = num_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, ax=ax, square=True, linewidths=0.5)
            ax.set_title("Feature correlations")
            st.pyplot(fig)
            plt.close()

    # --- Top categorical columns
    if not cat_df.empty:
        with st.expander("🏷️ Categorical Column Value Counts"):
            cat_cols_show = [c for c in cat_df.columns if df[c].nunique() <= 20][:4]
            if cat_cols_show:
                ncols = min(2, len(cat_cols_show))
                nrows = (len(cat_cols_show) + 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
                axes = np.array(axes).flatten() if len(cat_cols_show) > 1 else [axes]
                for i, col in enumerate(cat_cols_show):
                    vc = df[col].value_counts().head(15)
                    axes[i].barh(vc.index.astype(str), vc.values, color="#7F77DD")
                    axes[i].set_title(col, fontsize=11)
                    axes[i].invert_yaxis()
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No categorical columns with ≤20 unique values found.")

else:
    st.info("👆 Upload a CSV file to get started.")
