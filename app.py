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

    /* ======================================================= */
    /* STREAMLIT CLOUD PRO – PREMIUM UI THEME (DARK MODE)     */
    /* ======================================================= */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #0C111C;
        --bg2: #111827;
        --glass: rgba(255, 255, 255, 0.04);
        --glass2: rgba(255, 255, 255, 0.07);

        --text: #E2E8F0;
        --muted: #94A3B8;

        --primary: #4F8BFF;
        --primary2: #7AB6FF;
        --danger: #EF4444;
    }

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* ============================= */
    /* SIDEBAR – CLOUD PRO DESIGN   */
    /* ============================= */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A, #0A0F1D);
        border-right: 1px solid rgba(255,255,255,0.06);
        padding-top: 20px;
        box-shadow: 4px 0 20px rgba(0,0,0,0.25);
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 20px;
        padding-left: 6px;
    }

    /* Radio button styling */
    .stRadio > div { 
        gap: 12px;
    }

    /* ============================= */
    /* HEADER – CLOUD PRO BANNER    */
    /* ============================= */
    .header {
        padding: 18px 26px;
        background: linear-gradient(135deg, #2563EB, #3B82F6);
        border-radius: 14px;
        color: white !important;
        font-weight: 700;
        font-size: 26px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.45);
        margin-bottom: 25px;
        letter-spacing: 0.3px;
    }

    /* ============================= */
    /* CARD – GLASS CLOUD PRO       */
    /* ============================= */
    .card {
        background: var(--glass);
        padding: 26px;
        border-radius: 18px;
        backdrop-filter: blur(18px);
        border: 1px solid rgba(255,255,255,0.07);
        transition: 0.20s ease;
        box-shadow: 0 10px 28px rgba(0,0,0,0.4);
        margin-bottom: 18px;
    }

    .card:hover {
        transform: translateY(-3px);
        background: var(--glass2);
        box-shadow: 0 16px 40px rgba(0,0,0,0.55);
    }

    /* ============================= */
    /* SECTION TITLE                */
    /* ============================= */
    .section-title {
        font-size: 20px;
        color: var(--primary2);
        font-weight: 600;
        margin-bottom: 16px;
    }

    /* ============================= */
    /* BUTTON – ELEVATED PRO        */
    /* ============================= */
    .stButton > button {
        background: linear-gradient(135deg, #4F8BFF, #2563EB);
        border: none;
        padding: 10px 22px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        transition: 0.20s ease-in-out;
        box-shadow: 0 4px 14px rgba(0, 89, 255, 0.35);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #7AB6FF, #4F8BFF);
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 89, 255, 0.45);
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    /* ============================= */
    /* TABS – STREAMLIT PRO         */
    /* ============================= */
    div[data-baseweb="tab-list"] {
        gap: 10px !important;
        background: transparent !important;
        margin-bottom: 10px;
    }

    button[data-baseweb="tab"] {
        background: var(--glass);
        padding: 10px 18px;
        border-radius: 12px;
        color: var(--text);
        border: 1px solid rgba(255,255,255,0.07);
        transition: 0.20s ease;
    }

    button[data-baseweb="tab"]:hover {
        background: var(--glass2);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary);
        border-color: var(--primary2);
        font-weight: 600;
        color: white !important;
        box-shadow: 0 5px 14px rgba(0,89,255,0.45);
    }

    /* ============================= */
    /* INPUT FIELDS                  */
    /* ============================= */
    input, textarea, select {
        background: #1A2332 !important;
        border-radius: 10px !important;
        border: 1px solid #2C374C !important;
        color: var(--text) !important;
        padding: 6px 12px !important;
    }

    input:focus, select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }

    /* ============================= */
    /* DATAFRAME                    */
    /* ============================= */
    .stDataFrame div {
        color: var(--text) !important;
    }

    .stDataFrame table {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
    }

    thead tr th {
        background: rgba(255,255,255,0.05) !important;
        font-weight: 600 !important;
    }

    tbody tr:hover {
        background: rgba(255,255,255,0.06) !important;
    }

    /* ============================= */
    /* FILE UPLOADER                 */
    /* ============================= */
    .uploadedFile {
        background: #1A2332 !important;
        border-radius: 12px !important;
        padding: 12px;
        border: 1px dashed #334155 !important;
    }

    /* ============================= */
    /* SCROLLBAR (SEXY)              */
    /* ============================= */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 8px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
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
    - Menjelajah dokumentasi

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
                st.info(f"📊 Probabilitas: `{proba}`")

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
