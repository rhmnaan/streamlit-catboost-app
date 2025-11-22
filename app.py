import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# App config & Premium Styling
# ---------------------------
st.set_page_config(page_title="CatBoost - Spaceship Titanic", layout="wide")

def load_custom_css():
    st.markdown("""
    <style>
    :root { 
        --bg: #0d1117; 
        --card: #161b22; 
        --text: #e6edf3; 
        --muted: #8b949e; 
        --accent: #58a6ff; 
    }
    .stApp { background-color: var(--bg); color: var(--text); }

    .card {
        background: var(--card); 
        padding: 20px; 
        border-radius: 14px; 
        box-shadow: 0 8px 22px rgba(0,0,0,0.55);
        border: 1px solid #21262d;
    }
    .header {
        background: linear-gradient(90deg,#4c6ef5,#364fc7);
        padding: 18px; 
        border-radius: 12px; 
        color: white; 
        font-weight: 700; 
        font-size: 22px;
    }
    .section-title {
        color: var(--accent);
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
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

# --------------------------------------------------------
# Load Model
# --------------------------------------------------------
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# --------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------
st.sidebar.markdown("## ⚙️ Menu Navigasi")
page = st.sidebar.radio("Pilih Halaman:", 
                        ["Home", "Prediksi", "Analisis Data", "Dokumentasi / About"])

# --------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------
if page == "Home":
    st.markdown('<div class="header">🚀 CatBoost Spaceship Titanic</div>', unsafe_allow_html=True)
    st.write("---")

    st.markdown("""
    ## 💡 Tentang Project Ini  
    Project ini dibuat untuk memprediksi apakah penumpang Spaceship Titanic  
    akan **Transported** (dipindahkan ke dimensi lain) menggunakan model **CatBoost**.

    ### ✨ Fitur Utama  
    - Prediksi manual  
    - Prediksi banyak data via CSV  
    - Analisis data otomatis  
    - Tampilan premium dan modern  

    Aplikasi cocok untuk:  
    - Tugas kuliah  
    - Proyek machine learning  
    - Dashboard ML siap deploy  
    """)

# --------------------------------------------------------
# PAGE: PREDIKSI
# --------------------------------------------------------
elif page == "Prediksi":
    st.markdown('<div class="header">🧩 Prediksi Penumpang</div>', unsafe_allow_html=True)
    st.write("---")

    if not model_loaded:
        st.error(f"Model gagal dimuat: {model_error}")
    else:
        tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

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
                    Destination = st.selectbox("Destination",
                        ["TRAPPIST-1e","PSO J318.5-22","55 Cancri e","missing"])
                    Age = st.number_input("Age", 0, 120, 28)
                    VIP = st.selectbox("VIP", ["True","False","missing"])

                with col3:
                    RoomService = st.number_input("RoomService", 0.0)
                    FoodCourt = st.number_input("FoodCourt", 0.0)
                    ShoppingMall = st.number_input("ShoppingMall", 0.0)
                    Spa = st.number_input("Spa", 0.0)
                    VRDeck = st.number_input("VRDeck", 0.0)   # ✅ VRDECK TETAP ADA

                submit = st.form_submit_button("Prediksi Sekarang")

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
                    "VRDeck": VRDeck  # tetap diinput
                }

                df = pd.DataFrame([data])
                st.write("### 🔍 Data Input")
                st.dataframe(df)

                df_proc = simple_preprocess(df)
                df_proc = align_features(df_proc, model)

                pred = model.predict(df_proc)[0]
                proba = model.predict_proba(df_proc)[0]

                st.success(f"### Hasil Prediksi: **{bool(pred)}**")
                st.write("Probabilitas:", proba)

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------- CSV INPUT ----------------------
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Upload CSV</div>', unsafe_allow_html=True)

            file = st.file_uploader("Upload CSV", type="csv")

            if file:
                df = pd.read_csv(file)
                st.dataframe(df.head())

                if st.button("Prediksi CSV"):
                    df_proc = simple_preprocess(df)
                    df_proc = align_features(df_proc, model)

                    df["Transported"] = model.predict(df_proc).astype(bool)
                    st.success("Prediksi CSV berhasil!")

                    st.dataframe(df.head())

                    st.download_button(
                        "Download Hasil CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        "hasil_prediksi.csv"
                    )

            st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# ANALISIS DATA PAGE
# --------------------------------------------------------
elif page == "Analisis Data":
    st.markdown('<div class="header">📊 Analisis Data</div>', unsafe_allow_html=True)
    st.write("---")

    file = st.file_uploader("Upload CSV", type="csv")

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

# --------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------
else:
    st.markdown('<div class="header">📚 Dokumentasi</div>', unsafe_allow_html=True)
    st.write("---")

    st.markdown("""
    Aplikasi ini menggunakan model **CatBoost** untuk memprediksi apakah penumpang  
    akan *Transported* berdasarkan fitur-fitur penting.

    Jalankan aplikasi dengan:
    ```
    streamlit run app.py
    ```
    Pastikan file model:
    ```
    catboost_high_accuracy.cbm
    ```
    berada satu folder dengan `app.py`.
    """)
