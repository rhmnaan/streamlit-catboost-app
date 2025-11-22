import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import seaborn as sns
import shap
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
# MENU
# ==========================
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🧍 Prediksi Manual", "📁 Prediksi File CSV", "📊 Analisis Data", "📘 Dokumentasi Model"]
)

# =====================================================================
# HOME
# =====================================================================
if menu == "🏠 Home":
    st.markdown('<div class="main-title">🚀 Spaceship Titanic – CatBoost Prediction App</div>', 
                 unsafe_allow_html=True)

    st.markdown("""
    ### Aplikasi prediksi Spaceship Titanic menggunakan CatBoost  
    Fitur:
    - Prediksi manual
    - Prediksi via CSV
    - Analisis dataset
    - SHAP model interpretation
    """)

# =====================================================================
# 🧍 PREDIKSI MANUAL (INPUT SESUAI PERMINTAAN)
# =====================================================================
elif menu == "🧍 Prediksi Manual":
    st.markdown('<div class="main-title">🧍 Prediksi Penumpang (Input Manual)</div>', unsafe_allow_html=True)
    st.write("Masukkan data sesuai input berikut:")

    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            HomePlanet = st.selectbox("HomePlanet", ["Earth", "Mars", "Europa"])
            CryoSleep = st.selectbox("CryoSleep", ["True", "False"])
            Cabin = st.text_input("Cabin (misal B/0/P)")
            Destination = st.selectbox("Destination", [
                "TRAPPIST-1e",
                "55 Cancri e",
                "PSO J318.5-22"
            ])
            Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)

        with col2:
            VIP = st.selectbox("VIP", ["True", "False"])
            RoomService = st.number_input("RoomService", min_value=0.0, value=0.0)
            FoodCourt = st.number_input("FoodCourt", min_value=0.0, value=0.0)
            ShoppingMall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
            Spa = st.number_input("Spa", min_value=0.0, value=0.0)
            VRDeck = st.number_input("VRDeck", min_value=0.0, value=0.0)

        submitted = st.form_submit_button("Prediksi 🚀")

    if submitted:

        # UBAH BOOLEAN KE FORMAT ASLI
        CryoSleep_bool = True if CryoSleep == "True" else False
        VIP_bool = True if VIP == "True" else False

        # ==== SESUAIKAN DENGAN KOLOM TRAINING MODEL ====
        input_df = pd.DataFrame([{
            "HomePlanet": HomePlanet,
            "CryoSleep": CryoSleep_bool,
            "Cabin": Cabin,
            "Destination": Destination,
            "Age": Age,
            "VIP": VIP_bool,
            "RoomService": RoomService,
            "FoodCourt": FoodCourt,
            "ShoppingMall": ShoppingMall,
            "Spa": Spa,
            "VRDeck": VRDeck
        }])

        # ========== PREDIKSI ==========
        try:
            pred = model.predict(input_df)[0]
            st.success(f"🚀 Hasil Prediksi: **{bool(pred)}**")

        except Exception as e:
            st.error("❌ Error pada prediksi. Periksa kolom dan tipe data.")
            st.code(str(e))


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

        try:
            preds = model.predict(df)
            df["Transported"] = preds.astype(bool)

            st.write("### Hasil Prediksi")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Hasil CSV", csv, "prediction_output.csv")

        except Exception as e:
            st.error("❌ Error pada prediksi CSV.")
            st.code(str(e))

# =====================================================================
# 📊 ANALISIS DATA
# =====================================================================
elif menu == "📊 Analisis Data":
    st.markdown('<div class="main-title">📊 Analisis Data & SHAP Interpretation</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        # ========== Heatmap ==========
        st.subheader("📌 Correlation Heatmap")
        num_cols = df.select_dtypes(include=np.number).columns
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), ax=ax)
        st.pyplot(fig)

        # ========== SHAP ==========
        st.subheader("📌 SHAP Values")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df[num_cols].fillna(0).head(100))

        fig = shap.summary_plot(shap_values.values, df[num_cols].head(100), show=False)
        st.pyplot(bbox_inches="tight")

# =====================================================================
# 📘 DOKUMENTASI
# =====================================================================
elif menu == "📘 Dokumentasi Model":
    st.markdown('<div class="main-title">📘 Dokumentasi Model CatBoost</div>', unsafe_allow_html=True)

    st.write("""
    Model CatBoost dilatih dengan fitur:
    - HomePlanet
    - CryoSleep
    - Cabin
    - Destination
    - Age
    - VIP
    - RoomService
    - FoodCourt
    - ShoppingMall
    - Spa
    - VRDeck
    """)

