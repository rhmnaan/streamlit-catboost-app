# app.py
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

st.set_page_config(page_title="Spaceship Titanic - Prediksi Transported", page_icon="🚀", layout="centered")

st.title("🚀 Spaceship Titanic — Prediksi 'Transported' (CatBoost)")
st.markdown(
    "Masukkan nilai fitur di bawah ini (hanya fitur yang dipakai model). "
    "Default diset ke nilai umum; ubah sesuai kebutuhan."
)

# -----------------------
# PATH MODEL (sesuaikan jika perlu)
# -----------------------
# Jika Anda meletakkan model di folder models/, ubah jadi "models/catboost_high_accuracy.cbm"
MODEL_PATH = "/mnt/data/catboost_high_accuracy.cbm"

@st.cache_resource(show_spinner=False)
def load_model(path):
    model = CatBoostClassifier()
    model.load_model(path)
    return model

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model dari {MODEL_PATH}. Pastikan file ada. Error: {e}")
    st.stop()

# -----------------------
# Fungsi Preprocessing (mencerminkan pipeline Anda)
# -----------------------
def preprocess_input(df_raw):
    df = df_raw.copy()

    # Mapping kategori (sama seperti training)
    homeplanet_map = {'Earth': 1, 'Mars': 2, 'Europa': 3}
    destination_map = {'TRAPPIST-1e': 1, 'PSO J318.5-22': 2, '55 Cancri e': 3}
    deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    side_map = {'P': 1, 'S': 2}

    # Pastikan kolom ada
    for c in ["HomePlanet","Destination","Deck","Side","CryoSleep","VIP",
              "RoomService","FoodCourt","ShoppingMall","Spa","VRDeck",
              "Age","GroupID","GroupSize","IsAlone","CabinNum"]:
        if c not in df.columns:
            df[c] = 0

    # Mapping kategori → numerik
    df['HomePlanet'] = df['HomePlanet'].map(homeplanet_map).fillna(0).astype(int)
    df['Destination'] = df['Destination'].map(destination_map).fillna(0).astype(int)
    df['Deck'] = df['Deck'].map(deck_map).fillna(0).astype(int)
    df['Side'] = df['Side'].map(side_map).fillna(0).astype(int)

    # CryoSleep & VIP: menerima True/False/Unknown
    # Expect: values passed as 'True','False','Unknown' or boolean
    def map_bool_or_unknown(x):
        if pd.isna(x): return np.nan
        if isinstance(x, bool): return x
        s = str(x).strip().lower()
        if s in ["true","1","yes","y","t"]: return True
        if s in ["false","0","no","n","f"]: return False
        return np.nan

    df['CryoSleep_raw'] = df['CryoSleep'].apply(map_bool_or_unknown)
    df['VIP'] = df['VIP'].apply(map_bool_or_unknown).map({True:1, False:0}).fillna(0).astype(int)

    # Pengeluaran: treat as raw amounts; terapkan log1p sama seperti training
    spending_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for c in spending_cols:
        # jika ada nilai negatif, set 0; kemudian convert ke numeric
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        df[c] = df[c].clip(lower=0)  # tidak izinkan negatif
        df[c+'_log'] = np.log1p(df[c])  # transformasi log(1+x)

    # TotalSpendings: jumlah dari nilai ASLI (sebelum log)
    df['TotalSpendings'] = df[spending_cols].sum(axis=1)
    # log_TotalSpendings: gunakan log1p dari total (sama seperti training)
    df['log_TotalSpendings'] = np.log1p(df['TotalSpendings'])

    # CryoSleep_missing_flag dan imputasi berdasarkan aturan yang Anda pakai:
    # Jika CryoSleep missing dan TotalSpendings == 0 -> True
    # Jika CryoSleep missing dan TotalSpendings > 0 -> False
    df['CryoSleep_missing_flag'] = df['CryoSleep_raw'].isna().astype(int)
    df.loc[(df['CryoSleep_raw'].isna()) & (df['TotalSpendings'] == 0), 'CryoSleep_filled'] = True
    df.loc[(df['CryoSleep_raw'].isna()) & (df['TotalSpendings'] > 0), 'CryoSleep_filled'] = False
    df['CryoSleep_filled'] = df['CryoSleep_filled'].combine_first(df['CryoSleep_raw'])
    df['CryoSleep'] = df['CryoSleep_filled'].map({True:1, False:0}).fillna(0).astype(int)

    # Jika ada GroupSize/IsAlone/CabinNum pengguna masukkan langsung; pastikan tipe
    df['GroupSize'] = pd.to_numeric(df['GroupSize'], errors='coerce').fillna(1).astype(int)
    df['IsAlone'] = pd.to_numeric(df['IsAlone'], errors='coerce').fillna(0).astype(int)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce').fillna(0).astype(float)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0).astype(float)

    # Drop intermediate cols & sediakan final feature set (urutan bebas karena CatBoost memakai nama kolom)
    final_cols = [
        'HomePlanet','CryoSleep','Destination','Age','VIP',
        'RoomService','FoodCourt','ShoppingMall','Spa','VRDeck',
        'Deck','CabinNum','Side','GroupID','GroupSize','IsAlone',
        'TotalSpendings','log_TotalSpendings','CryoSleep_missing_flag'
    ]

    # Pastikan semua final cols ada
    for c in final_cols:
        if c not in df.columns:
            df[c] = 0

    # Gunakan kolom final (CatBoost akan mengenali berdasarkan nama kolom)
    df_final = df[final_cols].copy()

    # Untuk konsistensi: semua numeric
    df_final = df_final.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df_final

# -----------------------
# Form Input Streamlit (satu baris / single prediction)
# -----------------------
st.subheader("Form Input (satu baris)")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        homeplanet = st.selectbox("HomePlanet", ["Earth","Mars","Europa"])
        destination = st.selectbox("Destination", ["TRAPPIST-1e","PSO J318.5-22","55 Cancri e"])
        deck = st.selectbox("Deck", ["A","B","C","D","E","F","G","T"])
        side = st.selectbox("Side", ["P","S"])
        cabinnum = st.number_input("CabinNum (angka, 0 jika tidak tahu)", min_value=0.0, format="%.0f", value=0.0)
        groupid = st.text_input("GroupID (misal: '0001')", value="G1")
    with col2:
        cryo_option = st.selectbox("CryoSleep", ["True","False","Unknown"])
        vip_option = st.selectbox("VIP", ["False","True"])
        age = st.number_input("Age", min_value=0.0, value=30.0)
        room = st.number_input("RoomService (raw)", min_value=0.0, value=0.0, step=1.0)
        food = st.number_input("FoodCourt (raw)", min_value=0.0, value=0.0, step=1.0)
        shop = st.number_input("ShoppingMall (raw)", min_value=0.0, value=0.0, step=1.0)
        spa = st.number_input("Spa (raw)", min_value=0.0, value=0.0, step=1.0)
        vr = st.number_input("VRDeck (raw)", min_value=0.0, value=0.0, step=1.0)

    group_size = st.number_input("GroupSize (isi 1 jika sendiri)", min_value=1, value=1, step=1)
    is_alone = st.selectbox("IsAlone", [0,1], index=0, help="0 = bukan sendiri, 1 = sendiri")

    submitted = st.form_submit_button("Prediksi")

# Jika tombol ditekan -> proses
if submitted:
    # Susun input DataFrame
    cryo_val = True if cryo_option == "True" else (False if cryo_option == "False" else None)
    vip_val = True if vip_option == "True" else False

    input_dict = {
        "HomePlanet": homeplanet,
        "Destination": destination,
        "Deck": deck,
        "Side": side,
        "CabinNum": cabinnum,
        "GroupID": groupid,
        "GroupSize": group_size,
        "IsAlone": is_alone,
        "CryoSleep": cryo_val,
        "VIP": vip_val,
        "Age": age,
        "RoomService": room,
        "FoodCourt": food,
        "ShoppingMall": shop,
        "Spa": spa,
        "VRDeck": vr
    }

    input_df = pd.DataFrame([input_dict])
    processed = preprocess_input(input_df)

    # Prediksi menggunakan model
    try:
        pred = model.predict(processed)           # kelas (0/1)
        proba = model.predict_proba(processed)    # probabilitas tiap kelas
        prob_positive = float(proba[0][1]) if proba is not None else None

        label = int(pred[0])
        label_text = "Transported (True)" if label == 1 else "Not Transported (False)"

        st.markdown("### 🔮 Hasil Prediksi")
        st.write(f"**Label:** {label_text}")
        if prob_positive is not None:
            st.write(f"**Probabilitas (Transported):** {prob_positive:.4f}")
        st.json({
            "pred_label": int(label),
            "prob_transport": None if prob_positive is None else round(prob_positive, 6)
        })
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

# -----------------------
# Export / Batch predict (opsional)
# -----------------------
st.markdown("---")
st.subheader("Prediksi batch (unggah CSV)")
st.markdown("Unggah file CSV yang memiliki kolom input yang sama. Aplikasi akan menjalankan preprocessing yang sama dan mengeluarkan file hasil prediksi (CSV).")

uploaded = st.file_uploader("Pilih CSV (kolom sesuai fitur input)", type=["csv"])
if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
        st.write("Contoh 5 baris dari file Anda:")
        st.dataframe(df_upload.head())

        if st.button("Jalankan Prediksi Batch"):
            df_proc = preprocess_input(df_upload)
            preds = model.predict(df_proc)
            probs = model.predict_proba(df_proc)[:,1] if hasattr(model, "predict_proba") else [None]*len(preds)

            res = df_upload.copy()
            res["Pred_Transported"] = preds.astype(int)
            res["Prob_Transported"] = probs
            out_name = "predictions_result.csv"
            res.to_csv(out_name, index=False)
            st.success("Prediksi batch selesai.")
            st.download_button("Unduh hasil prediksi (CSV)", res.to_csv(index=False).encode('utf-8'), file_name=out_name, mime="text/csv")
    except Exception as e:
        st.error(f"Error membaca/menjalankan prediksi pada file: {e}")

# -----------------------
# Footer: instruksi singkat deploy
# -----------------------
st.markdown("---")
st.markdown("**Instruksi singkat menjalankan lokal:**\n\n"
            "1. Pastikan Python environment Anda memiliki paket: `streamlit`, `pandas`, `numpy`, `catboost`.\n"
            "2. Simpan file ini sebagai `app.py` dan letakkan model `catboost_high_accuracy.cbm` pada `MODEL_PATH`.\n"
            "3. Jalankan: `streamlit run app.py`.\n\n"
            "**Untuk deploy ke Streamlit Cloud:**\n"
            " - Buat repo GitHub berisi `app.py` dan folder `models/catboost_high_accuracy.cbm` (jika model terlalu besar, gunakan hosting file seperti Google Drive dan ubah MODEL_PATH atau gunakan `st.secrets` untuk link). "
            " - Hubungkan repo ke Streamlit Cloud dan deploy.")
