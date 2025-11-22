import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# App config & CSS styling
# ---------------------------
st.set_page_config(page_title="CatBoost - Spaceship Titanic", layout="wide")

def load_custom_css():
    st.markdown("""
    <style>
    :root { --bg: #0e1117; --card: #0f1720; --muted: #9aa4b2; --accent: #58a6ff; }
    .stApp { background-color: var(--bg); color: #e6eef8; }
    .card { background: var(--card); padding: 18px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); border: 1px solid #1f2933; }
    .header { background: linear-gradient(90deg,#4b79a1,#283e51); padding: 18px; border-radius: 12px; color: white; font-weight:700; }
    .section-title { color: var(--accent); font-weight:600; margin-bottom:8px; }
    .muted { color: var(--muted); }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ---------------------------
# Helper functions
# ---------------------------
MODEL_PATH = "catboost_high_accuracy.cbm"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}.")
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
            df_proc[c] = df_proc[c].astype(str)

    return df_proc

def get_model_feature_names(model):
    try:
        if hasattr(model, 'feature_names_'):
            return list(model.feature_names_)
    except:
        pass
    try:
        return list(model.get_feature_names())
    except:
        pass
    return None

def align_features(df: pd.DataFrame, model):
    expected = get_model_feature_names(model)
    if expected is None:
        st.warning("Model tidak menyediakan daftar fitur.")
        return df

    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = np.nan

    df = df[expected].copy()
    return df

# ---------------------------
# Load model
# ---------------------------
try:
    model = load_model()
    model_loaded = True
    MODEL_FEATURE_NAMES = get_model_feature_names(model)
except Exception as e:
    model_loaded = False
    model = None
    MODEL_FEATURE_NAMES = None
    model_error = str(e)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.markdown("# ⚙️ Menu")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Prediksi", "Analisis Data", "Dokumentasi / About"])
st.sidebar.markdown("---")

if st.sidebar.checkbox("Tampilkan info model", value=False):
    if model_loaded:
        st.sidebar.write("Model Loaded: ✔️")
        st.sidebar.write(MODEL_FEATURE_NAMES)
    else:
        st.sidebar.error(model_error)

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.markdown('<div class="header">🚀 CatBoost Spaceship Titanic</div>', unsafe_allow_html=True)
    st.write("---")
    st.markdown("""
    ## Fitur Aplikasi  
    - Prediksi manual (1 penumpang)  
    - Upload CSV untuk prediksi banyak data  
    - Visualisasi distribusi fitur  
    - Dokumentasi lengkap  
    """)

# ---------------------------
# Prediksi Page
# ---------------------------
elif page == "Prediksi":
    st.markdown('<div class="header">🧩 Prediksi</div>', unsafe_allow_html=True)
    st.write("---")

    if not model_loaded:
        st.error("Model tidak ditemukan.")
    else:
        tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

        # --------------------- Manual ---------------------
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Input Manual</div>', unsafe_allow_html=True)

            with st.form("manual_form"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    HomePlanet = st.selectbox('HomePlanet', ['Earth','Europa','Mars','missing'])
                    CryoSleep = st.selectbox('CryoSleep', ['True','False','missing'])
                    Cabin = st.text_input('Cabin', value='B/0/0')

                with col2:
                    Destination = st.selectbox('Destination', [
                        'TRAPPIST-1e','PSO J318.5-22','55 Cancri e','missing'
                    ])
                    Age = st.number_input('Age', 0.0, 120.0, 30.0)
                    VIP = st.selectbox('VIP', ['True','False','missing'])

                with col3:
                    RoomService = st.number_input('RoomService', 0.0, value=0.0)
                    FoodCourt = st.number_input('FoodCourt', 0.0, value=0.0)
                    ShoppingMall = st.number_input('ShoppingMall', 0.0, value=0.0)
                    Spa = st.number_input('Spa', 0.0, value=0.0)
                    VRDeck = st.number_input('VRDeck', 0.0, value=0.0)

                submit = st.form_submit_button("Prediksi")

            if submit:
                df = pd.DataFrame([{
                    'HomePlanet': HomePlanet,
                    'CryoSleep': CryoSleep,
                    'Cabin': Cabin,
                    'Destination': Destination,
                    'Age': Age,
                    'VIP': VIP,
                    'RoomService': RoomService,
                    'FoodCourt': FoodCourt,
                    'ShoppingMall': ShoppingMall,
                    'Spa': Spa,
                    'VRDeck': VRDeck
                }])

                st.write("Preview Data")
                st.dataframe(df)

                df_proc = simple_preprocess(df)
                df_proc = align_features(df_proc, model)

                try:
                    pred = model.predict(df_proc)[0]
                    proba = model.predict_proba(df_proc)[0]
                    st.success(f"Hasil Prediksi: **{bool(pred)}**")
                    st.write("Probabilitas:", proba)
                except Exception as e:
                    st.error(f"Error prediksi: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

        # --------------------- CSV ---------------------
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Upload CSV</div>', unsafe_allow_html=True)

            file = st.file_uploader("Upload File CSV", type="csv")

            if file:
                df = pd.read_csv(file)
                st.dataframe(df.head())

                if st.button("Prediksi CSV"):
                    df_proc = simple_preprocess(df)
                    df_proc = align_features(df_proc, model)

                    try:
                        df["Transported"] = model.predict(df_proc).astype(bool)
                        st.success("Berhasil memprediksi!")
                        st.dataframe(df.head())

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Hasil CSV", csv, "prediksi_catboost.csv")
                    except Exception as e:
                        st.error(f"Error prediksi CSV: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Analisis Data Page
# ---------------------------
elif page == "Analisis Data":
    st.markdown('<div class="header">📊 Analisis Data</div>', unsafe_allow_html=True)
    st.write("---")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        st.write(df.describe(include="all"))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        col = st.selectbox("Pilih Kolom", numeric_cols + cat_cols)

        fig, ax = plt.subplots()

        if col in numeric_cols:
            ax.hist(df[col].dropna(), bins=30)
            ax.set_title(f"Distribusi {col}")
        else:
            vc = df[col].value_counts().head(20)
            ax.bar(vc.index, vc.values)
            ax.set_title(f"Frekuensi {col}")
            plt.xticks(rotation=45)

        st.pyplot(fig)

# ---------------------------
# About Page
# ---------------------------
elif page == "Dokumentasi / About":
    st.markdown('<div class="header">📚 Dokumentasi</div>', unsafe_allow_html=True)
    st.write("---")
    st.markdown("""
    Aplikasi ini digunakan untuk memprediksi *Transported* pada dataset **Spaceship Titanic** menggunakan **CatBoost**.
    
    ### Cara Menggunakan
    1. Pastikan file `catboost_high_accuracy.cbm` berada di folder yang sama.
    2. Jalankan aplikasi:
       ```
       streamlit run app.py
       ```
    3. Pilih menu:
       - Prediksi manual  
       - Prediksi CSV  
       - Analisis data  
    """)

