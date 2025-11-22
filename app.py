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
# FUNGSI PREPROCESSING SESUAI TRAINING
# ==========================

def extract_cabin_features(cabin):
    try:
        deck, num, side = cabin.split("/")
        return deck, int(num), side
    except:
        return None, None, None

def preprocess_input(df):

    # Cabin = Deck / CabinNum / Side
    df["Deck"], df["CabinNum"], df["Side"] = zip(*df["Cabin"].apply(extract_cabin_features))

    # GroupID → ambil angka dari CabinNum jika ada, kalau tidak → random
    df["GroupID"] = df["CabinNum"].fillna(0).astype(int)

    # GroupSize → 1 (karena hanya input manual)
    df["GroupSize"] = 1

    # IsAlone
    df["IsAlone"] = df["GroupSize"].apply(lambda x: 1 if x == 1 else 0)

    # TotalSpendings
    df["TotalSpendings"] = (
        df["RoomService"] +
        df["FoodCourt"] +
        df["ShoppingMall"] +
        df["Spa"] +
        df["VRDeck"]
    )

    # log_TotalSpendings
    df["log_TotalSpendings"] = np.log1p(df["TotalSpendings"])

    # CryoSleep_missing_flag
    df["CryoSleep_missing_flag"] = df["CryoSleep"].isna().astype(int)

    # Age_Group
    def age_group(age):
        if age < 12:
            return "Child"
        elif age < 18:
            return "Teen"
        elif age < 30:
            return "YoungAdult"
        elif age < 60:
            return "Adult"
        else:
            return "Senior"

    df["Age_Group"] = df["Age"].apply(age_group)

    return df


# ==========================
# MENU
# ==========================
menu = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🧍 Prediksi Manual", "📁 Prediksi File CSV"]
)

# =====================================================================
# HOME
# =====================================================================
if menu == "🏠 Home":
    st.markdown('<div class="main-title">🚀 Spaceship Titanic – CatBoost Prediction App</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Aplikasi prediksi Spaceship Titanic menggunakan CatBoost  
    Fitur:
    - Prediksi manual
    - Prediksi via CSV
    """)

# =====================================================================
# 🧍 PREDIKSI MANUAL
# =====================================================================
elif menu == "🧍 Prediksi Manual":
    st.markdown('<div class="main-title">🧍 Prediksi Penumpang (Input Manual)</div>', unsafe_allow_html=True)

    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            HomePlanet = st.selectbox("HomePlanet", ["Earth", "Mars", "Europa"])
            CryoSleep = st.selectbox("CryoSleep", ["True", "False"])
            Cabin = st.text_input("Cabin (misal: B/0/P)")
            Destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
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

        input_df = pd.DataFrame([{
            "HomePlanet": HomePlanet,
            "CryoSleep": True if CryoSleep == "True" else False,
            "Cabin": Cabin,
            "Destination": Destination,
            "Age": Age,
            "VIP": True if VIP == "True" else False,
            "RoomService": RoomService,
            "FoodCourt": FoodCourt,
            "ShoppingMall": ShoppingMall,
            "Spa": Spa,
            "VRDeck": VRDeck
        }])

        # Preprocessing otomatis
        input_df = preprocess_input(input_df)

        try:
            pred = model.predict(input_df)[0]
            st.success(f"🚀 Hasil Prediksi: **{bool(pred)}**")
        except Exception as e:
            st.error("❌ Error pada prediksi.")
            st.code(str(e))

# =====================================================================
# 📁 PREDIKSI CSV
# =====================================================================
elif menu == "📁 Prediksi File CSV":
    st.markdown('<div class="main-title">📁 Prediksi Banyak Data (CSV)</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df = preprocess_input(df)

        try:
            preds = model.predict(df)
            df["Transported"] = preds.astype(bool)

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Hasil CSV", csv, "prediction_output.csv")

        except Exception as e:
            st.error("❌ Error prediksi CSV.")
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

