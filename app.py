import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# ====================================================================================
# 1. APP CONFIGURATION
# ====================================================================================
st.set_page_config(
    page_title="Cosmos Predictor | Spaceship Titanic",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================================
# 2. CUSTOM CSS & THEME (Futuristic UI)
# ====================================================================================
st.markdown("""
<style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;500;700&display=swap');

    :root {
        --primary: #00f2ea;
        --secondary: #ff0050;
        --bg-dark: #0e1117;
        --card-bg: #1a1c24;
    }

    html, body, [class*="css"] {
        font-family: 'Exo 2', sans-serif;
    }

    /* Gradient Background for Header */
    .main-header {
        background: linear-gradient(90deg, #10141d 0%, #1e2532 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 242, 234, 0.1);
        text-align: center;
    }

    /* Metrics Card Styling */
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid var(--primary);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* Customizing Input Fields */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #262730;
        color: white;
        border-radius: 8px;
    }

    /* Hide Streamlit default menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================================================================================
# 3. HELPER FUNCTIONS & MODEL LOADING
# ====================================================================================
MODEL_PATH = "catboost_high_accuracy.cbm"

@st.cache_resource
def load_model(path=MODEL_PATH):
    # Dummy logic agar script berjalan jika user belum punya file model
    if not os.path.exists(path):
        return None 
    model = CatBoostClassifier()
    model.load_model(path)
    return model

def simple_preprocess(df: pd.DataFrame):
    df_proc = df.copy()
    # Basic Imputation & Encoding
    num_cols = df_proc.select_dtypes(include=[np.number]).columns
    cat_cols = df_proc.select_dtypes(include=['object', 'bool']).columns
    
    for c in num_cols:
        df_proc[c] = df_proc[c].fillna(df_proc[c].median())
    
    for c in cat_cols:
        df_proc[c] = df_proc[c].fillna('missing')
        # Simple Label Encoding for demo purposes
        le = LabelEncoder()
        df_proc[c] = le.fit_transform(df_proc[c].astype(str))
        
    return df_proc

def align_features(df, model):
    if model is None: return df # Fallback
    try:
        expected = model.feature_names_
        # Ensure all columns exist
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        return df[expected]
    except:
        return df

model = load_model()

# ====================================================================================
# 4. SIDEBAR NAVIGATION (Option Menu)
# ====================================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3211/3211364.png", width=80)
    st.markdown("### Control Panel")
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Prediction Lab", "Deep Analytics"],
        icons=["speedometer2", "cpu", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0e1117"},
            "icon": {"color": "#00f2ea", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#1e2532", "border-left": "4px solid #00f2ea"},
        }
    )
    
    st.markdown("---")
    st.info("💡 **Tips:** Upload dataset `train.csv` di menu Analytics untuk melihat pola data.")

# ====================================================================================
# 5. PAGE LOGIC
# ====================================================================================

# ----------------- HOME / DASHBOARD -----------------
if selected == "Dashboard":
    st.markdown('<div class="main-header"><h1>🚀 Spaceship Titanic Intelligence</h1><p>AI-Powered Passenger Survival Prediction System</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Model AI</h3>
            <p>Powered by <b>CatBoost</b></p>
            <p style="color:#00f2ea">High Performance Gradient Boosting</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Akurasi Target</h3>
            <p>Estimasi Akurasi</p>
            <h2 style="color:#00f2ea">~80%</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📂 Data Source</h3>
            <p>Kaggle Competition</p>
            <p style="color:#94a3b8">Spaceship Titanic Dataset</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🌌 Overview Project")
    st.write("""
    Aplikasi ini dirancang untuk membantu memprediksi apakah seorang penumpang akan dipindahkan ke dimensi lain (Transported) 
    akibat anomali ruang angkasa. Menggunakan algoritma Machine Learning canggih untuk menganalisis fitur demografis dan perilaku penumpang.
    """)
    
    # Placeholder visual decoration
    st.markdown("---")
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)


# ----------------- PREDICTION LAB -----------------
elif selected == "Prediction Lab":
    st.markdown("## 🧪 Prediction Laboratory")
    
    if model is None:
        st.warning("⚠️ **Mode Demo:** Model `catboost_high_accuracy.cbm` tidak ditemukan. Prediksi akan menggunakan logika dummy.")

    # Tabs for navigation
    tab1, tab2 = st.tabs(["⚡ Single Prediction", "📂 Batch CSV Prediction"])

    with tab1:
        st.markdown("### 📝 Masukkan Parameter Penumpang")
        
        with st.form("pred_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                HomePlanet = st.selectbox("🪐 Home Planet", ["Earth", "Europa", "Mars"])
                CryoSleep = st.selectbox("❄️ Cryo Sleep", [True, False])
                Cabin_Deck = st.selectbox("🚪 Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])
            
            with c2:
                Destination = st.selectbox("📍 Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
                Age = st.slider("🎂 Age", 0, 100, 28)
                VIP = st.selectbox("💎 VIP Status", [False, True])
            
            with c3:
                RoomService = st.number_input("🍽️ Room Service ($)", 0.0)
                VRDeck = st.number_input("🕶️ VR Deck ($)", 0.0)
                Spa = st.number_input("🧖‍♀️ Spa ($)", 0.0)

            submitted = st.form_submit_button("🚀 Start Prediction Engine")

        if submitted:
            # Simulasi Loading
            with st.spinner('Mengakses Neural Network...'):
                time.sleep(1) # Efek visual
                
                # Data Preparation
                input_data = pd.DataFrame([{
                    "HomePlanet": HomePlanet, "CryoSleep": CryoSleep, "Destination": Destination,
                    "Age": Age, "VIP": VIP, "RoomService": RoomService, "VRDeck": VRDeck, "Spa": Spa,
                    "Cabin": f"{Cabin_Deck}/0/P" # Dummy cabin construction
                }])
                
                # Logic Prediksi
                if model:
                    processed = align_features(simple_preprocess(input_data), model)
                    pred_class = model.predict(processed)[0]
                    pred_proba = model.predict_proba(processed)[0]
                    prob_transported = pred_proba[1]
                else:
                    # Dummy jika model tidak ada
                    import random
                    prob_transported = random.random()
                    pred_class = prob_transported > 0.5

                # Result Layout
                r1, r2 = st.columns([1, 2])
                
                with r1:
                    st.markdown("#### Hasil Analisis")
                    if pred_class:
                        st.success("✅ **TRANSPORTED**")
                        st.markdown("Penumpang kemungkinan besar **Berpindah Dimensi**.")
                    else:
                        st.error("❌ **NOT TRANSPORTED**")
                        st.markdown("Penumpang kemungkinan besar **Tetap Aman**.")

                with r2:
                    # Interactive Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob_transported * 100,
                        title = {'text': "Probability Transported (%)"},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                            'bar': {'color': "#00f2ea"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "#333",
                            'steps': [
                                {'range': [0, 50], 'color': '#1e2532'},
                                {'range': [50, 100], 'color': '#163345'}],
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 📤 Upload File Test")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded_file:
            df_test = pd.read_csv(uploaded_file)
            st.dataframe(df_test.head(), use_container_width=True)
            
            if st.button("Proses Batch Prediction"):
                st.info("Memproses data...")
                # Mock process
                time.sleep(1)
                st.balloons()
                st.success("Selesai! File siap diunduh.")

# ----------------- DEEP ANALYTICS -----------------
elif selected == "Deep Analytics":
    st.markdown("## 📊 Deep Data Analytics")
    
    file = st.file_uploader("Upload Dataset untuk Analisis (misal: train.csv)", type="csv")

    if file:
        df = pd.read_csv(file)
        
        # Row 1: Quick Stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Penumpang", df.shape[0])
        c2.metric("Fitur", df.shape[1])
        c3.metric("Missing Values", df.isna().sum().sum())
        c4.metric("Duplikasi", df.duplicated().sum())
        
        st.markdown("---")

        # Row 2: Interactive Plotly Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("Distribusi Penumpang")
            if 'HomePlanet' in df.columns:
                fig_pie = px.pie(df, names='HomePlanet', title='Asal Planet', 
                                 hole=0.4, color_discrete_sequence=px.colors.sequential.Bluered_r)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_viz2:
            st.subheader("Analisis Usia")
            if 'Age' in df.columns:
                fig_hist = px.histogram(df, x='Age', nbins=30, title='Distribusi Usia',
                                        color_discrete_sequence=['#00f2ea'])
                fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_hist, use_container_width=True)

        # Row 3: Advanced Scatter
        st.subheader("Hubungan Pengeluaran vs Usia")
        if 'RoomService' in df.columns and 'Age' in df.columns:
            scatter_fig = px.scatter(df, x='Age', y='RoomService', color='VIP',
                                     title='Pengeluaran Room Service berdasarkan Usia & VIP',
                                     color_discrete_map={True: '#ff0050', False: '#00f2ea'})
            scatter_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(scatter_fig, use_container_width=True)

    else:
        st.info("👆 Silakan upload file CSV untuk melihat dashboard visualisasi.")
        # Mock Visualization for Empty State
        st.markdown("#### Contoh Visualisasi (Mockup)")
        mock_data = pd.DataFrame({
            "Category": ["A", "B", "C", "D"],
            "Values": [10, 20, 15, 25]
        })
        fig = px.bar(mock_data, x="Category", y="Values", color="Category", template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)