import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------
# APP CONFIG + PREMIUM DARK MODE
# -------------------------------------------------------------------
st.set_page_config(page_title="CatBoost Spaceship Titanic", layout="wide")

def load_custom_css():
    st.markdown("""
    <style>

    /* GLOBAL COLOR VARIABLES */
    :root { 
        --bg: #0d1117; 
        --card: #161b22; 
        --card-hover: #1c2330;
        --text: #e6edf3; 
        --muted: #8b949e; 
        --accent: #58a6ff;
    }

    .stApp { background-color: var(--bg); color: var(--text); }

    /* HEADER STYLE */
    .header {
        background: linear-gradient(90deg, #3b82f6, #1e40af);
        padding: 18px; 
        border-radius: 14px; 
        color: white; 
        font-weight: 700; 
        text-align: left;
        font-size: 23px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }

    /* CARD STYLE */
    .card {
        background: var(--card); 
        padding: 22px; 
        border-radius: 16px; 
        border: 1px solid #21262d;
        box-shadow: 0 10px 25px rgba(0,0,0,0.45);
        transition: all 0.2s ease;
    }
    .card:hover {
        background: var(--card-hover);
        box-shadow: 0 14px 32px rgba(0,0,0,0.55);
    }

    /* SECTION TITLE */
    .section-title {
        color: var(--accent);
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 10px;
    }

    /* SIDEBAR */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #11161d !important;
        border-right: 1px solid #1f2937 !important;
        color: var(--text) !important;
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: var(--accent);
        padding-bottom: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------
MODEL_PATH = "catboost_high_accuracy.cbm"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan di: {path}")
    model = CatBoostClassifier()
    model.load_model(path)
    return model

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
    if expected is None:
        return df

    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = np.nan

    return df[expected]

# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# -------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Menu Navigasi</div>", unsafe_allow_html=True)

page = st.sidebar.radio("Pilih Halaman:", 
                        ["Home", "Prediksi", "Analisis Data"])

# -------------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------------
if page == "Home":
    st.markdown('<div class="header">🚀 CatBoost Spaceship Titanic</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 👋 Selamat datang di Dashboard 
    Aplikasi ini menggunakan model **CatBoost** berakurasi tinggi untuk memprediksi  
    apakah penumpang *Spaceship Titanic* akan **Transported**.

    ---
    #### ✨ Apa yang bisa kamu lakukan?
    - Melakukan prediksi dari satu data penumpang  
    - Melakukan prediksi massal dari file CSV  
    - Melihat analisis data  

    ⚡ By : Kelompok 3
    """)

# -------------------------------------------------------------------
# PAGE: PREDIKSI
# -------------------------------------------------------------------
elif page == "Prediksi":
    st.markdown('<div class="header">🧩 Prediksi Penumpang</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Model gagal dimuat: {model_error}")
    else:
        tab1, tab2 = st.tabs(["📥 Input Manual", "📤 Upload CSV"])

        # ---------------------- MANUAL INPUT ----------------------
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
                    Destination = st.selectbox(
                        "Destination",
                        ["TRAPPIST-1e","PSO J318.5-22","55 Cancri e","missing"]
                    )
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
                data = {
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
                }

                df = pd.DataFrame([data])
                st.write("### 📄 Data Input")
                st.dataframe(df)

                df_proc = simple_preprocess(df)
                df_proc = align_features(df_proc, model)

                pred = model.predict(df_proc)[0]
                proba = model.predict_proba(df_proc)[0]

                st.success(f"### 🎯 Hasil Prediksi: **{bool(pred)}**")
                st.info(f"📊 Probabilitas: {proba}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------- CSV INPUT ----------------------
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Upload CSV</div>', unsafe_allow_html=True)

            file = st.file_uploader("Upload file test.csv", type="csv")

            if file:
                df = pd.read_csv(file)
                st.dataframe(df.head())

                if st.button("🚀 Prediksi CSV"):
                    df_proc = simple_preprocess(df)
                    df_proc = align_features(df_proc, model)

                    df["Transported"] = model.predict(df_proc).astype(bool)
                    st.success("Prediksi CSV selesai!")

                    st.dataframe(df.head())

                    st.download_button(
                        "Download Hasil CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        "hasil_prediksi.csv"
                    )

            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# ANALISIS DATA PAGE
# -------------------------------------------------------------------
elif page == "Analisis Data":
    st.markdown('<div class="header">📊 Analisis Data</div>', unsafe_allow_html=True)

    file = st.file_uploader("Upload file test.csv", type="csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=['object']).columns.tolist()

        col = st.selectbox("Pilih Kolom", numeric + categorical)

        fig, ax = plt.subplots()

        if col in numeric:
            ax.hist(df[col].dropna(), bins=30)
        else:
            vc = df[col].value_counts().head(20)
            ax.bar(vc.index, vc.values)
            plt.xticks(rotation=45)

        st.pyplot(fig)
