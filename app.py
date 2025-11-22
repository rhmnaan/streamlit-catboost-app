"""
Streamlit Premium Multipage App
Features:
- Home
- Predict (Manual input + CSV upload)
- Analysis (distribution plots)
- SHAP explanations
- About / Documentation

Simpan file ini sebagai `app.py`. Pastikan file model CatBoost `catboost_high_accuracy.cbm` berada di folder yang sama.

Requirements (requirements.txt):
streamlit
pandas
numpy
catboost
matplotlib
shap
scikit-learn

Note: SHAP dapat memakan waktu dan memerlukan instalasi paket `shap`.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Try to import shap; if not available, we'll provide graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------
# App config & CSS styling
# ---------------------------
st.set_page_config(page_title="CatBoost - Spaceship Titanic (Premium)", layout="wide")

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
        raise FileNotFoundError(f"Model not found at {path}. Please upload catboost model to repo root.")
    model = CatBoostClassifier()
    model.load_model(path)
    return model

@st.cache_data
def load_sample_data():
    # small helper: return empty template or sample
    return None

def simple_preprocess(df: pd.DataFrame, fit_label_encoders: bool=False):
    """Very small preprocessing: fillna, encode simple categorical objects with LabelEncoder.
    This is a heuristic — for best results, apply the same preprocessing used in training."""
    df_proc = df.copy()
    # Normalize column names: ensure no leading/trailing spaces
    df_proc.columns = df_proc.columns.str.strip()

    # Fill numeric NA with column median
    for c in df_proc.select_dtypes(include=[np.number]).columns:
        try:
            median = df_proc[c].median()
        except Exception:
            median = 0
        df_proc[c] = df_proc[c].fillna(median)

    # Fill object NA with 'missing' and simple label encode
    for c in df_proc.select_dtypes(include=['object','category']).columns:
        df_proc[c] = df_proc[c].fillna('missing')
        le = LabelEncoder()
        try:
            df_proc[c] = le.fit_transform(df_proc[c].astype(str))
        except Exception:
            df_proc[c] = df_proc[c].astype(str)

    # Convert bool columns to int (CatBoost handles bool, but keep consistent)
    for c in df_proc.select_dtypes(include=['bool']).columns:
        df_proc[c] = df_proc[c].astype(int)

    return df_proc

def get_model_feature_names(cat_model):
    """
    Try multiple ways to extract feature names from CatBoost model object.
    Returns list of feature names or None if not found.
    """
    try:
        if hasattr(cat_model, 'feature_names_'):
            fn = getattr(cat_model, 'feature_names_')
            if fn is not None:
                return list(fn)
    except Exception:
        pass

    try:
        if hasattr(cat_model, 'get_feature_names'):
            try:
                fn = cat_model.get_feature_names()
                if fn is not None:
                    return list(fn)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if hasattr(cat_model, 'feature_names'):
            fn = getattr(cat_model, 'feature_names')
            if fn is not None:
                return list(fn)
    except Exception:
        pass

    try:
        if hasattr(cat_model, 'get_params'):
            params = cat_model.get_params()
            if isinstance(params, dict) and 'feature_names' in params:
                return list(params['feature_names'])
    except Exception:
        pass

    return None

def align_features(df: pd.DataFrame, model):
    """
    Ensure df has columns in the same names & order expected by the model.
    - If model exposes expected feature names, add missing columns with NaN and drop extras.
    - If not, returns df unchanged but warns user.
    """
    if model is None:
        # No model loaded, nothing to align
        return df

    expected = None
    try:
        expected = get_model_feature_names(model)
    except Exception:
        expected = None

    if expected is None:
        # can't determine expected names
        st.warning("Tidak dapat membaca daftar fitur dari model. Pastikan kolom input sesuai urutan yang dipakai saat training.")
        return df

    # Add missing columns with NaN
    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = np.nan

    # Drop extras not expected by model
    extra = [c for c in df.columns if c not in expected]
    if extra:
        df = df.drop(columns=extra)

    # Reorder to expected
    df = df[expected].copy()

    return df

# ---------------------------
# Load model (on startup)
# ---------------------------
try:
    model = load_model()
    model_loaded = True
    try:
        MODEL_FEATURE_NAMES = get_model_feature_names(model)
    except Exception:
        MODEL_FEATURE_NAMES = None
except Exception as e:
    model_loaded = False
    model = None
    model_load_error = str(e)
    MODEL_FEATURE_NAMES = None

# ---------------------------
# App layout (sidebar navigation)
# ---------------------------
st.sidebar.markdown("# ⚙️ Menu")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Prediksi", "Analisis Data", "SHAP", "Dokumentasi / About"])
st.sidebar.markdown("---")
if st.sidebar.checkbox("Tampilkan info model", value=False):
    if model_loaded:
        st.sidebar.write("Model: catboost")
        st.sidebar.write(f"Model file: {MODEL_PATH}")
        if MODEL_FEATURE_NAMES is not None:
            st.sidebar.write("Contoh fitur yang diharapkan (preview):")
            st.sidebar.write(MODEL_FEATURE_NAMES[:100])
    else:
        st.sidebar.error("Model belum dimuat. Pastikan file .cbm ada di repo root.")

# ---------------------------
# Home page
# ---------------------------
if page == "Home":
    st.markdown('<div class="header">🚀 CatBoost Spaceship Titanic - Premium App</div>', unsafe_allow_html=True)
    st.write("---")
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("""
        ## Ringkasan
        Aplikasi ini menyediakan:
        - Input manual untuk 1 penumpang (prediksi langsung)
        - Upload CSV untuk prediksi batch
        - Visualisasi distribusi fitur
        - Penjelasan model menggunakan SHAP (jika tersedia)
        """)
        st.write("Untuk deploy, pastikan `catboost_high_accuracy.cbm` berada pada root repository.")
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**Quick tips**')
        if model_loaded:
            st.write('Model status: ✅ loaded')
        else:
            st.warning('Model status: ❌ not loaded')
            st.caption(model_load_error if 'model_load_error' in globals() else '')
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Prediksi page
# ---------------------------
elif page == "Prediksi":
    st.markdown('<div class="header">🧩 Prediksi</div>', unsafe_allow_html=True)
    st.write("---")
    if not model_loaded:
        st.error("Model tidak tersedia. Unggah file model `catboost_high_accuracy.cbm` ke folder aplikasi.")
    # Two tabs: Manual input and Upload CSV
    tab1, tab2 = st.tabs(["Manual Input","Upload & Batch"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Manual - 1 Penumpang</div>', unsafe_allow_html=True)
        st.write('Masukkan fitur sesuai dataset. Jika model mengharapkan fitur lain, tambahkan di CSV atau ubah form.')

        # Example form fields — diperluas agar tidak missing common features (Spa, VRDeck, dll.)
        with st.form(key='manual_form'):
            colA, colB, colC = st.columns(3)
            with colA:
                HomePlanet = st.selectbox('HomePlanet', ['Earth','Europa','Mars','missing'])
                CryoSleep = st.selectbox('CryoSleep', ['True','False','missing'])
                Cabin = st.text_input('Cabin (as string)', value='B/0/0')
                Deck = st.text_input('Deck (optional)', value='')
            with colB:
                Destination = st.selectbox('Destination', ['TRAPPIST-1e','PSO J318.5-22','55 Cancri e','missing'])
                Age = st.number_input('Age', min_value=0.0, max_value=120.0, value=30.0)
                VIP = st.selectbox('VIP', ['True','False','missing'])
                Side = st.text_input('Side (optional)', value='')
            with colC:
                RoomService = st.number_input('RoomService', min_value=0.0, value=0.0)
                FoodCourt = st.number_input('FoodCourt', min_value=0.0, value=0.0)
                ShoppingMall = st.number_input('ShoppingMall', min_value=0.0, value=0.0)
                Spa = st.number_input('Spa', min_value=0.0, value=0.0)
                VRDeck = st.number_input('VRDeck', min_value=0.0, value=0.0)

            submit_manual = st.form_submit_button('Predict')

        if submit_manual:
            # Build dataframe with same columns as training (best effort)
            df_manual = pd.DataFrame([{
                'HomePlanet': HomePlanet,
                'CryoSleep': CryoSleep,
                'Cabin': Cabin,
                'Deck': Deck,
                'Destination': Destination,
                'Age': Age,
                'VIP': VIP,
                'Side': Side,
                'RoomService': RoomService,
                'FoodCourt': FoodCourt,
                'ShoppingMall': ShoppingMall,
                'Spa': Spa,
                'VRDeck': VRDeck
            }])

            st.write('Preview input:')
            st.dataframe(df_manual)

            # Preprocess
            df_proc = simple_preprocess(df_manual)

            # Align features to what model expects
            try:
                df_proc = align_features(df_proc, model)
            except Exception as e:
                st.error(f'Gagal menyelaraskan fitur dengan model: {e}')

            # Predict
            try:
                preds = model.predict(df_proc)
                probs = None
                try:
                    probs = model.predict_proba(df_proc)
                except Exception:
                    probs = None
                transported = bool(preds[0])
                st.success(f"Hasil prediksi: Transported = {transported}")
                if probs is not None:
                    st.write('Probabilitas (label order depends on model):')
                    st.write(probs)
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload CSV untuk Prediksi Batch</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader('Upload file CSV (test.csv)', type=['csv'])
        show_rows = st.selectbox('Preview rows', ['5','10','All'])

        if uploaded is not None:
            df_test = pd.read_csv(uploaded)
            if show_rows == '5':
                st.dataframe(df_test.head())
            elif show_rows == '10':
                st.dataframe(df_test.head(10))
            else:
                st.dataframe(df_test)

            if st.button('Run Batch Prediction'):
                with st.spinner('Running predictions...'):
                    df_proc = simple_preprocess(df_test)
                    try:
                        df_proc = align_features(df_proc, model)
                    except Exception as e:
                        st.warning('Warning saat menyelaraskan fitur: ' + str(e))
                    try:
                        preds = model.predict(df_proc)
                        df_out = df_test.copy()
                        df_out['Transported'] = preds.astype(bool)
                        st.success('Selesai! Tabel hasil di bawah:')
                        st.dataframe(df_out.head(20))

                        csv = df_out.to_csv(index=False).encode('utf-8')
                        st.download_button('Download CSV hasil', data=csv, file_name='submission_catboost.csv')
                    except Exception as e:
                        st.error(f'Error saat prediksi batch: {e}')
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Analysis page
# ---------------------------
elif page == "Analisis Data":
    st.markdown('<div class="header">📊 Analisis Data</div>', unsafe_allow_html=True)
    st.write('---')
    uploaded = st.file_uploader('Upload CSV (untuk analisis)', type=['csv'])
    if uploaded is None:
        st.info('Unggah dataset untuk melihat grafik distribusi fitur.')
    else:
        df = pd.read_csv(uploaded)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Statistik Singkat</div>', unsafe_allow_html=True)
        st.write(df.describe(include='all'))
        st.markdown('</div>', unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Pilih Fitur untuk Visualisasi</div>', unsafe_allow_html=True)
        col = st.selectbox('Pilih kolom', numeric_cols + cat_cols if (numeric_cols+cat_cols) else [None])
        if col:
            st.markdown('</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8,4))
            if col in numeric_cols:
                ax.hist(df[col].dropna(), bins=30)
                ax.set_title(f'Distribusi: {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
            else:
                vc = df[col].value_counts().head(20)
                ax.bar(vc.index.astype(str), vc.values)
                plt.xticks(rotation=45, ha='right')
                ax.set_title(f'Frekuensi: {col}')
            st.pyplot(fig)

        # Quick pair plot limited
        if st.checkbox('Tampilkan pairplot (terbatas)'):
            st.write('Pairplot may be slow for large datasets')

            sample = df.sample(n=min(200, len(df)))
            sns = None
            try:
                import seaborn as sns
                sns.pairplot(
                    sample.select_dtypes(include=[np.number]).dropna(axis=1),
                    diag_kind='kde'
                )
                st.pyplot(plt.gcf())
            except Exception as e:
                st.error('Seaborn tidak tersedia atau terjadi error saat membuat pairplot: ' + str(e))


# ---------------------------
# SHAP explanation page
# ---------------------------
elif page == "SHAP":
    st.markdown('<div class="header">🔬 SHAP - Penjelasan Model</div>', unsafe_allow_html=True)
    st.write('---')
    if not model_loaded:
        st.error('Model tidak dimuat; SHAP memerlukan model untuk dijelaskan.')
    elif not SHAP_AVAILABLE:
        st.warning('Paket `shap` tidak terpasang. Install shap di requirements.txt untuk mengaktifkan fitur ini.')
    else:
        st.info('SHAP akan menghitung kontribusi fitur. Ini mungkin memakan waktu.')
        uploaded = st.file_uploader('Upload CSV (digunakan sebagai background / sample)', type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Pilih sampel untuk SHAP</div>', unsafe_allow_html=True)
            sample_n = st.slider('Jumlah sampel background untuk explainer', min_value=10, max_value=1000, value=100, step=10)
            sample_df = df.sample(n=min(sample_n, len(df)))
            st.write('Menggunakan sample background berukuran', len(sample_df))

            # preprocess & align
            X = simple_preprocess(sample_df)
            try:
                X = align_features(X, model)
            except Exception:
                st.warning('Gagal menyelaraskan fitur untuk SHAP sample. Melanjutkan dengan kolom yang ada.')

            explainer = shap.TreeExplainer(model)
            with st.spinner('Menghitung SHAP values...'):
                shap_values = explainer.shap_values(X)

            st.success('SHAP values selesai dihitung.')

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Global Feature Importance (mean |SHAP|)</div>', unsafe_allow_html=True)
            try:
                abs_mean = np.abs(shap_values).mean(axis=0)
            except Exception:
                # if shap_values is list for multiclass
                abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            feat_imp = pd.DataFrame({ 'feature': X.columns, 'mean_abs_shap': abs_mean })
            feat_imp = feat_imp.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
            st.dataframe(feat_imp.head(30))

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">SHAP force plot (1 sample)</div>', unsafe_allow_html=True)
            idx = st.number_input('Pilih index sample (0..n-1)', min_value=0, max_value=len(X)-1, value=0)
            # show force plot
            try:
                # handle shap outputs gracefully
                if isinstance(shap_values, list):
                    # multiclass: shap_values is list (n_classes x n_samples x n_features)
                    sv = shap_values[0]  # use first class for visualization (user may change)
                    shap_html = shap.force_plot(explainer.expected_value[0], sv[idx,:], X.iloc[idx,:], matplotlib=False)
                else:
                    shap_html = shap.force_plot(explainer.expected_value, shap_values[idx,:], X.iloc[idx,:], matplotlib=False)
                st.components.v1.html(shap_html.html(), height=400)
            except Exception as e:
                st.error('Gagal menampilkan force plot: '+str(e))
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Documentation / About
# ---------------------------
elif page == "Dokumentasi / About":
    st.markdown('<div class="header">📚 Dokumentasi & About</div>', unsafe_allow_html=True)
    st.write('---')
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('**Tentang Aplikasi**')
    st.write('Aplikasi ini dibuat untuk mendemonstrasikan deploy model CatBoost pada Streamlit dengan fitur premium.')
    st.markdown('**Cara pakai**')
    st.markdown('1. Pastikan `catboost_high_accuracy.cbm` ada di root repo.\n2. Deploy ke Streamlit Cloud atau jalankan `streamlit run app.py`.')
    st.markdown('**Catatan penting**')
    st.markdown('- Preprocessing yang dipakai di sini bersifat generik. Untuk hasil terbaik, gunakan preprocessing yang sama seperti saat training model.')
    st.markdown('- SHAP membutuhkan paket `shap` dan dapat lambat untuk dataset besar.')
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# End of file
# ---------------------------
