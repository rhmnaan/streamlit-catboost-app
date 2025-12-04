import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ====================================================================================
# APP CONFIG
# ====================================================================================
st.set_page_config(page_title="CatBoost Spaceship Titanic", layout="wide")

# ====================================================================================
# CUSTOM CSS PREMIUM
# ====================================================================================
def load_custom_css():
    st.markdown("""
    <style>

    /* ===================================================== */
    /* GLOBAL THEME */
    /* ===================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --bg: #0b0f19;
        --bg2: #121826;
        --card: rgba(255, 255, 255, 0.05);
        --card2: rgba(255, 255, 255, 0.1);
        --text: #e6edf3;
        --muted: #94a3b8;
        --accent: #3b82f6;
        --accent2: #60a5fa;
    }

    html, body, .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
        animation: fadePage 0.6s ease;
    }

    @keyframes fadePage {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    /* ===================================================== */
    /* HEADER */
    /* ===================================================== */
    .header {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        padding: 22px 28px;
        border-radius: 18px;
        color: white;
        font-weight: 700;
        font-size: 26px;
        letter-spacing: 0.3px;
        margin-bottom: 25px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.55);
        animation: fadeDown 0.6s ease;
    }

    @keyframes fadeDown {
        from {opacity: 0; transform: translateY(-15px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* ===================================================== */
    /* CARDS */
    /* ===================================================== */
    .card {
        background: var(--card);
        padding: 25px;
        border-radius: 20px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        transition: 0.25s;
        animation: fadeUp 0.7s ease;
    }

    .card:hover {
        background: var(--card2);
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.55);
    }

    @keyframes fadeUp {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .section-title {
        font-size: 19px;
        font-weight: 600;
        color: var(--accent2);
        margin-bottom: 12px;
    }

    /* ===================================================== */
    /* SIDEBAR */
    /* ===================================================== */
    section[data-testid="stSidebar"] {
        background: #0f1522;
        border-right: 1px solid #1e293b;
        padding: 15px;
        animation: fadePage 0.5s ease;
    }

    .sidebar-title {
        font-size: 23px;
        font-weight: bold;
        color: var(--accent);
        padding-top: 10px;
        padding-bottom: 15px;
    }

    /* ===================================================== */
    /* BUTTON */
    /* ===================================================== */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white !important;
        padding: 10px 22px;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        transition: 0.25s;
        box-shadow: 0 5px 15px rgba(0,0,0,0.4);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        transform: translateY(-3px);
        box-shadow: 0 8px 18px rgba(0,0,0,0.5);
    }

    /* ===================================================== */
    /* INPUT FIELD */
    /* ===================================================== */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1b2333 !important;
        border-radius: 10px !important;
        border: 1px solid #273244 !important;
        color: var(--text) !important;
    }

    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ====================================================================================
# MODEL LOADING
# ====================================================================================
MODEL_PATH = "catboost_high_accuracy.cbm"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError("⚠️ File model tidak ditemukan!")
    model = CatBoostClassifier()
    model.load_model(path)
    return model

# ====================================================================================
# PREPROCESS
# ====================================================================================
def simple_preprocess(df: pd.DataFrame):
    df_proc = df.copy()
    df_proc.columns = df_proc.columns.str.strip()

    for c in df_proc.select_dtypes(include=[np.number]).columns:
        df_proc[c] = df_proc[c].fillna(df_proc[c].median())

    for c in df_proc.select_dtypes(include=['object','category']).columns:
        df_proc[c] = df_proc[c].fillna('missing')
        le = LabelEncoder()
        try:
            df_proc[c] = le.fit_transform(df_proc[c].astype(str))
        except:
            pass

    return df_proc

def get_model_feature_names(model):
    try:
        return list(model.feature_names_)
    except:
        try:
            return list(model.get_feature_names())
        except:
            return None

def align_features(df, model):
    expected = get_model_feature_names(model)
    if expected:
        for col in expected:
            if col not in df:
                df[col] = np.nan
        return df[expected]
    return df

# ====================================================================================
# LOAD MODEL
# ====================================================================================
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ====================================================================================
# SIDEBAR NAVIGATION
# ====================================================================================
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Menu Navigasi</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Prediksi", "Analisis Data"])

# ====================================================================================
# HOME PAGE
# ====================================================================================
if page == "Home":
    st.markdown('<div class="header">🚀 CatBoost Spaceship Titanic</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 👋 Selamat datang!
    Dashboard ini memungkinkan kamu melakukan:
    - 🔮 Prediksi penumpang (manual & CSV)
    - 📊 Analisis data dinamis
    - 📈 Visualisasi statistik interaktif  
    ---
    """)

# ====================================================================================
# PREDIKSI PAGE
# ====================================================================================
elif page == "Prediksi":
    st.markdown('<div class="header">🧩 Prediksi Penumpang</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"❌ Model gagal dimuat: {model_error}")
    else:
        tab1, tab2 = st.tabs(["📥 Input Manual", "📤 Upload CSV"])

        # ----------------------------------------------------
        # TAB 1 – INPUT MANUAL
        # ----------------------------------------------------
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Isi Data Penumpang</div>', unsafe_allow_html=True)

            with st.form("form_manual"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    HomePlanet = st.selectbox("HomePlanet", ["Earth","Europa","Mars","missing"])
                    CryoSleep = st.selectbox("CryoSleep", ["True","False","missing"])
                    Cabin = st.text_input("Cabin", "B/0/0")

                with col2:
                    Destination = st.selectbox("Destination", ["TRAPPIST-1e","PSO J318.5-22","55 Cancri e","missing"])
                    Age = st.number_input("Age", 0, 120, 28)
                    VIP = st.selectbox("VIP", ["True","False","missing"])

                with col3:
                    RoomService = st.number_input("RoomService", 0.0)
                    FoodCourt = st.number_input("FoodCourt", 0.0)
                    ShoppingMall = st.number_input("ShoppingMall", 0.0)
                    Spa = st.number_input("Spa", 0.0)
                    VRDeck = st.number_input("VRDeck", 0.0)

                submit = st.form_submit_button("🔮 Prediksi Sekarang")

            if submit:
                df = pd.DataFrame([{
                    "HomePlanet": HomePlanet,
                    "CryoSleep": CryoSleep,
                    "Cabin": Cabin,
                    "Destination": Destination,
                    "Age": Age,
                    "VIP": VIP,
                    "RoomService": RoomService,
                    "FoodCourt": FoodCourt,
                    "ShoppingMall": ShoppingMall,
                    "Spa": Spa,
                    "VRDeck": VRDeck
                }])

                st.write("### 📄 Data Input")
                st.dataframe(df)

                df_proc = align_features(simple_preprocess(df), model)

                pred = model.predict(df_proc)[0]
                proba = model.predict_proba(df_proc)[0]

                st.success(f"### 🎯 Prediksi: **{bool(pred)}**")
                st.info(f"📊 Probabilitas: `{proba}`")

            st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # TAB 2 – UPLOAD CSV
        # ----------------------------------------------------
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Upload CSV</div>', unsafe_allow_html=True)

            file = st.file_uploader("Upload file test.csv", type="csv")

            if file:
                df = pd.read_csv(file)
                st.dataframe(df.head())

                if st.button("🚀 Prediksi CSV"):
                    df_proc = align_features(simple_preprocess(df), model)
                    df["Transported"] = model.predict(df_proc).astype(bool)

                    st.success("✔️ Prediksi selesai!")
                    st.dataframe(df.head())

                    st.download_button(
                        "Download Hasil CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        "hasil_prediksi.csv"
                    )

            st.markdown("</div>", unsafe_allow_html=True)

# ====================================================================================
# ANALISIS DATA PAGE
# ====================================================================================
elif page == "Analisis Data":
    st.markdown('<div class="header">📊 Analisis Data</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=['object']).columns.tolist()

        col = st.selectbox("Pilih Kolom untuk Visualisasi", numeric + categorical)

        fig, ax = plt.subplots(figsize=(8, 4))

        if col in numeric:
            ax.hist(df[col].dropna(), bins=28)
            ax.set_title(f"Distribusi {col}")
        else:
            vc = df[col].value_counts().head(20)
            ax.bar(vc.index, vc.values)
            ax.set_xticklabels(vc.index, rotation=45, ha='right')

        st.pyplot(fig)

