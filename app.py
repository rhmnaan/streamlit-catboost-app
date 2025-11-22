import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# CONFIGURASI APLIKASI
# ==========================
st.set_page_config(
    page_title="Spaceship Titanic - CatBoost App",
    page_icon="🚀",
    layout="wide"
)

# ==========================
# CSS PREMIUM
# ==========================
st.markdown("""
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 800;
            text-align: center;
            color: white;
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            margin-bottom: 30px;
        }
        .card {
            padding: 20px;
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
MODEL_PATH = "catboost_model.cbm"

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# ==========================
# HALAMAN UTAMA (MULTIPAGE)
# ==========================
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🧍 Prediksi Manual", "📁 Prediksi File CSV", "📊 Analisis Data", "📘 Dokumentasi Model"]
)

# =====================================================================
# 🏠 HOME
# =====================================================================
if menu == "🏠 Home":
    st.markdown('<div class="main-title">🚀 Spaceship Titanic – CatBoost Prediction App</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Selamat datang di aplikasi prediksi **Spaceship Titanic**  
    Aplikasi ini dibangun menggunakan **Streamlit** dan **CatBoostClassifier** dengan fitur:
    - Prediksi data **satuan**
    - Prediksi **banyak data (CSV)**
    - Visualisasi distribusi fitur
    - **SHAP interpretation** untuk menjelaskan model
    - Pairplot, heatmap, dan grafik data lainnya  
    """)

# =====================================================================
# 🧍 PREDIKSI MANUAL
# =====================================================================
elif menu == "🧍 Prediksi Manual":
    st.markdown('<div class="main-title">🧍 Prediksi Penumpang (Input Manual)</div>', unsafe_allow_html=True)
    st.write("Masukkan data di bawah untuk memprediksi apakah penumpang *Transported* atau tidak.")

    with st.container():
        with st.form("manual_form"):
            col1, col2 = st.columns(2)

            with col1:
                CryoSleep = st.selectbox("CryoSleep", [0, 1])
                VIP = st.selectbox("VIP", [0, 1])
                RoomService = st.number_input("RoomService", 0, 20000, 0)
                FoodCourt = st.number_input("FoodCourt", 0, 20000, 0)

            with col2:
                ShoppingMall = st.number_input("ShoppingMall", 0, 20000, 0)
                Spa = st.number_input("Spa", 0, 20000, 0)
                VRDeck = st.number_input("VRDeck", 0, 20000, 0)

            submitted = st.form_submit_button("Prediksi 🚀")

    if submitted:
        input_df = pd.DataFrame([{
            "CryoSleep": CryoSleep,
            "VIP": VIP,
            "RoomService": RoomService,
            "FoodCourt": FoodCourt,
            "ShoppingMall": ShoppingMall,
            "Spa": Spa,
            "VRDeck": VRDeck,
        }])

        pred = model.predict(input_df)[0]
        st.success(f"Hasil Prediksi: **{bool(pred)}**")

# =====================================================================
# 📁 PREDIKSI CSV
# =====================================================================
elif menu == "📁 Prediksi File CSV":
    st.markdown('<div class="main-title">📁 Prediksi Banyak Data (CSV)</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("### Data yang diupload")
        st.dataframe(df.head())

        preds = model.predict(df)
        df["Transported"] = preds.astype(bool)

        st.write("### Hasil Prediksi")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Hasil CSV", csv, "prediction_output.csv")

# =====================================================================
# 📊 ANALISIS DATA
# =====================================================================
elif menu == "📊 Analisis Data":
    st.markdown('<div class="main-title">📊 Analisis Data & SHAP Interpretation</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload dataset untuk analisis", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("### Data")
        st.dataframe(df.head())

        # ---- Distribusi ---
        st.subheader("📌 Distribusi Fitur Numerik")
        num_cols = df.select_dtypes(include=np.number).columns
        
        fig, ax = plt.subplots(figsize=(12, 5))
        df[num_cols].hist(ax=ax)
        st.pyplot(fig)

        # ---- Heatmap ---
        st.subheader("📌 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=False, ax=ax)
        st.pyplot(fig)

        # ---- Pairplot ---
        st.subheader("📌 Pairplot (Sampel 200 Data)")
        if st.checkbox("Tampilkan pairplot"):
            st.info("Pairplot bisa lambat pada dataset besar.")
            sample = df.sample(min(200, len(df)))
            fig = sns.pairplot(sample[num_cols])
            st.pyplot(fig)

        # ---- SHAP ---
        st.subheader("📌 SHAP Model Interpretation")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df[num_cols].head(100))

        st.write("### SHAP Summary Plot")
        fig = shap.summary_plot(shap_values, df[num_cols].head(100), show=False)
        st.pyplot(bbox_inches='tight')

# =====================================================================
# 📘 DOKUMENTASI
# =====================================================================
elif menu == "📘 Dokumentasi Model":
    st.markdown('<div class="main-title">📘 Dokumentasi Model CatBoost</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Model yang digunakan:
    - **CatBoostClassifier**
    - Optimized untuk dataset **Spaceship Titanic**
    - Mampu menangani data numerik & kategorikal
    - Mendukung interpretasi via **SHAP values**

    ### Fitur Aplikasi:
    - Prediksi penumpang (manual & CSV)
    - Heatmap, distribusi, histogram
    - Pairplot
    - SHAP Interpretation
    """)

