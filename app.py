import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# ---------- CONFIG ----------
st.set_page_config(page_title="Spaceship Titanic - Manual Predictor", layout="wide")
st.title("🚀 Spaceship Titanic - Prediksi Manual (CatBoost)")

# ---------- PATH MODEL ----------
MODEL_PATH = "/mnt/data/catboost_model.cbm"

@st.cache_resource
def load_model(path=MODEL_PATH):
    m = CatBoostClassifier()
    m.load_model(path)
    return m

model = load_model()

# ---------- Mapping sesuai pipeline training kamu ----------
homeplanet_map = {'Earth': 1, 'Mars': 2, 'Europa': 3}
destination_map = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
side_map = {'P': 1, 'S': 2}

# Age group bins used in training (labels 1..3)
def age_group_label(age):
    # bins = [0,17,59,inf] labels=[1,2,3]
    if age <= 17:
        return 1
    elif age <= 59:
        return 2
    else:
        return 3

# ---------- Helper preprocessing function (mirip pipeline training) ----------
def preprocess_manual_input(df_raw):
    """
    df_raw: dataframe berisi kolom:
      HomePlanet (str), CryoSleep (0/1), Cabin (str A/0/P), Destination (str),
      Age (float), VIP (0/1), RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
    Menghasilkan df siap predict dengan semua fitur training.
    """
    df = df_raw.copy()

    # --- Split Cabin into Deck, CabinNum, Side (if format invalid -> fill -1)
    def split_cabin(c):
        try:
            parts = str(c).split("/")
            deck = parts[0] if len(parts) > 0 else ""
            num = parts[1] if len(parts) > 1 else ""
            side = parts[2] if len(parts) > 2 else ""
            deck_num = deck_map.get(deck, 0)  # training used 1..8, use 0 for unknown
            side_num = side_map.get(side, 0)  # 1/2, use 0 for unknown
            cabinnum = int(num) if str(num).isdigit() else 0
            return deck_num, cabinnum, side_num
        except Exception:
            return 0, 0, 0

    deck_vals = []
    cabinnum_vals = []
    side_vals = []
    for c in df['Cabin'].astype(str).values:
        d, n, s = split_cabin(c)
        deck_vals.append(d)
        cabinnum_vals.append(n)
        side_vals.append(s)
    df['Deck'] = deck_vals
    df['CabinNum'] = cabinnum_vals
    df['Side'] = side_vals

    # --- Map categories to numeric consistent with training
    df['HomePlanet'] = df['HomePlanet'].map(homeplanet_map).fillna(0).astype(int)
    df['Destination'] = df['Destination'].map(destination_map).fillna(0).astype(int)

    # CryoSleep / VIP already provided as 0/1 - ensure int
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    df['VIP'] = df['VIP'].astype(int)

    # --- Group features (training computed from PassengerId; for manual input set default)
    # If PassengerId exists use it, else set GroupID=0
    if 'PassengerId' in df.columns:
        df['GroupID'] = df['PassengerId'].astype(str).str.split("_").str[0].astype(float)
    else:
        df['GroupID'] = 0.0
    # For single manual input, GroupSize = 1
    df['GroupSize'] = 1
    df['IsAlone'] = 1  # since GroupSize==1

    # --- TotalSpendings then log transform (training did log1p on spendings)
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spend_cols:
        if col not in df.columns:
            df[col] = 0.0
        # ensure numeric
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df['TotalSpendings'] = df[spend_cols].sum(axis=1)
    # Training applied log1p to spendings (they did transform individual spend columns and TotalSpendings)
    # We'll apply log1p to each spending column and to TotalSpendings to match training normalization
    for col in spend_cols + ['TotalSpendings']:
        df[col] = np.log1p(df[col])

    # --- Age_Group according to training bins mapping (1/2/3)
    df['Age_Group'] = df['Age'].apply(lambda x: age_group_label(float(x) if pd.notna(x) else 0)).astype(int)

    # --- CryoSleep_missing_flag (manual input provided => flag 0)
    df['CryoSleep_missing_flag'] = 0

    # Ensure numeric dtypes
    numeric_cols = [
        'HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt',
        'ShoppingMall','Spa','VRDeck','Deck','CabinNum','Side','GroupID','GroupSize',
        'IsAlone','TotalSpendings','log_TotalSpendings' # log_TotalSpendings may be same as TotalSpendings here
    ]
    # Note: model feature list contains 'log_TotalSpendings' but we computed TotalSpendings (log1p)
    # To be safe, create 'log_TotalSpendings' same as TotalSpendings (since training applied log1p to total)
    df['log_TotalSpendings'] = df['TotalSpendings']

    # Cast numeric types and fill NaN with 0
    for c in df.columns:
        if c in ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck',
                 'TotalSpendings','log_TotalSpendings','GroupID','CabinNum','Deck','Side','Age_Group']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Finally, reorder columns to match model.feature_names_ if available
    try:
        feature_order = model.feature_names_
        # ensure all model features exist in df, otherwise create with zeros
        for f in feature_order:
            if f not in df.columns:
                df[f] = 0
        df = df[feature_order]
    except Exception:
        # fallback: keep current df
        pass

    return df

# ---------- Instruction / keterangan di atas form ----------
st.markdown("""
**Keterangan format input (Harus diikuti):**
- `HomePlanet`: pilih dari Earth / Mars / Europa (akan otomatis di-encode menjadi angka sesuai training).  
- `CryoSleep` dan `VIP`: pilih True/False (akan di-convert ke 1/0).  
- `Cabin`: format `Deck/Num/Side` contoh `B/23/P`. Jika format berbeda, nilai default akan dipakai.  
- `Destination`: pilih dari TRAPPIST-1e / 55 Cancri e / PSO J318.5-22 (otomatis di-encode).  
- `Age`: masukkan angka (contoh: 25) — **Age harus angka**.  
- `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`: masukkan angka (0 jika tidak ada).  
""")

st.markdown("---")

# ---------- Input Form ----------
with st.form("manual_form"):
    col1, col2 = st.columns(2)
    with col1:
        HomePlanet = st.selectbox("HomePlanet", ["Earth","Mars","Europa"])
        CryoSleep = st.selectbox("CryoSleep", ["True","False"])
        Cabin = st.text_input("Cabin (format Deck/Num/Side)", value="B/0/P")
        Destination = st.selectbox("Destination", ["TRAPPIST-1e","55 Cancri e","PSO J318.5-22"])
        Age = st.number_input("Age (angka)", min_value=0.0, max_value=120.0, value=30.0)
    with col2:
        VIP = st.selectbox("VIP", ["True","False"])
        RoomService = st.number_input("RoomService", min_value=0.0, value=0.0)
        FoodCourt = st.number_input("FoodCourt", min_value=0.0, value=0.0)
        ShoppingMall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
        Spa = st.number_input("Spa", min_value=0.0, value=0.0)
        VRDeck = st.number_input("VRDeck", min_value=0.0, value=0.0)

    submit = st.form_submit_button("Prediksi 🚀")

# ---------- On Submit ----------
if submit:
    # Build raw DF
    raw = pd.DataFrame([{
        "HomePlanet": HomePlanet,
        "CryoSleep": 1 if CryoSleep == "True" else 0,
        "Cabin": Cabin,
        "Destination": Destination,
        "Age": Age,
        "VIP": 1 if VIP == "True" else 0,
        "RoomService": RoomService,
        "FoodCourt": FoodCourt,
        "ShoppingMall": ShoppingMall,
        "Spa": Spa,
        "VRDeck": VRDeck
    }])

    # Preprocess to match training pipeline
    X = preprocess_manual_input(raw)

    # Final check: ensure no strings remain
    non_numeric = [c for c in X.columns if X[c].dtype == object]
    if non_numeric:
        st.warning(f"Kolom non-numeric terdeteksi (diubah ke 0): {non_numeric}")
        for c in non_numeric:
            X[c] = 0

    # Debug: tampilkan dataframe yang akan dipakai model (opsional)
    st.subheader("Data input (setelah preprocessing, cocokkan fitur model)")
    st.dataframe(X)

    # Predict
    try:
        preds = model.predict(X)
        probs = None
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None

        transported = bool(preds[0])
        st.success(f"✅ Hasil Prediksi: Transported = {transported}")
        if probs is not None:
            st.info(f"Probabilitas (kelas 1): {probs[0][-1]:.4f}")
    except Exception as e:
        st.error("❌ Error saat prediksi. Pastikan input sudah sesuai format dan semua kolom numeric.")
        st.code(str(e))
