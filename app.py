import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_high_accuracy.cbm")
    return model

model = load_model()

st.title("🚀 Prediksi Status Penumpang - Streamlit")

# ------------------------------------
# INPUT YANG INGIN DIBERIKAN USER
# ------------------------------------
st.subheader("Masukkan Data Penumpang (Tidak harus lengkap)")

homeplanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars", None], index=0)
destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e", None])
age = st.number_input("Age", min_value=0, max_value=100, step=1, value=25)
vip = st.selectbox("VIP", ["True", "False", None])
cryosleep = st.selectbox("CryoSleep", ["True", "False", None])

room_service = st.number_input("RoomService", min_value=0, step=1, value=0)
food_court = st.number_input("FoodCourt", min_value=0, step=1, value=0)
shopping = st.number_input("ShoppingMall", min_value=0, step=1, value=0)

# ------------------------------------
# BENTUKKAN DICTIONARY INPUT
# ------------------------------------
data_dict = {
    "HomePlanet": homeplanet,
    "CryoSleep": cryosleep,
    "Destination": destination,
    "Age": age,
    "VIP": vip,
    "RoomService": room_service,
    "FoodCourt": food_court,
    "ShoppingMall": shopping,
}

df = pd.DataFrame([data_dict])

# ------------------------------------
# PREDIKSI
# ------------------------------------
if st.button("Prediksi"):
    try:
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        st.success(f"✅ Hasil Prediksi: **{pred}**")
        st.info(f"Probabilitas: {proba:.4f}")

    except Exception as e:
        st.error("❌ Terjadi error saat memprediksi")
        st.code(str(e))
