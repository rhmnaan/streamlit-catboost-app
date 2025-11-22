import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

st.set_page_config(page_title="Spaceship Titanic", page_icon="🚀", layout="wide")

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "catboost_model.cbm"

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

model = load_model()


# -----------------------------
# MAPPING SESUAI TRAINING
# -----------------------------
map_HomePlanet = {"Earth": 0, "Mars": 1, "Europa": 2}
map_Destination = {"TRAPPIST-1e": 0, "55 Cancri e": 1, "PSO J318.5-22": 2}
map_Deck = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
map_Side = {"P": 0, "S": 1}

# Age Group mapping
def age_group(age):
    if age < 12: return 0
    elif age < 18: return 1
    elif age < 30: return 2
    elif age < 60: return 3
    else: return 4


# -----------------------------
# PREPROCESS SESUAI MODEL
# -----------------------------
def preprocess(df):

    # ---- Cabin Split ----
    def split_cabin(c):
        try:
            deck, num, side = c.split("/")
            deck = map_Deck.get(deck, -1)
            side = map_Side.get(side, -1)
            num = int(num)
            return deck, num, side
        except:
            return -1, -1, -1

    df["Deck"], df["CabinNum"], df["Side"] = zip(*df["Cabin"].apply(split_cabin))

    # ---- Convert Category → Numeric ----
    df["HomePlanet"] = df["HomePlanet"].map(map_HomePlanet)
    df["Destination"] = df["Destination"].map(map_Destination)

    df["CryoSleep"] = df["CryoSleep"].astype(int)
    df["VIP"] = df["VIP"].astype(int)

    # ---- Group Features ----
    df["GroupID"] = df["CabinNum"].fillna(0)
    df["GroupSize"] = 1
    df["IsAlone"] = 1

    # ---- Total Spending ----
    df["TotalSpendings"] = (
        df["RoomService"] +
        df["FoodCourt"] +
        df["ShoppingMall"] +
        df["Spa"] +
        df["VRDeck"]
    )

    df["log_TotalSpendings"] = np.log1p(df["TotalSpendings"])

    # ---- Age Group ----
    df["Age_Group"] = df["Age"].apply(age_group)

    # ---- Cryo Missing Flag ----
    df["CryoSleep_missing_flag"] = 0

    return df


# -----------------------------
# UI PREDIKSI
# -----------------------------
st.title("🚀 Prediksi Penumpang Spaceship Titanic")

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        HomePlanet = st.selectbox("HomePlanet (otomatis diubah ke angka)", ["Earth", "Mars", "Europa"])
        CryoSleep = st.selectbox("CryoSleep (True = 1, False = 0)", ["True", "False"])
        Cabin = st.text_input("Cabin (format: A/0/P)")
        Destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
        Age = st.number_input("Age (Tetap Angka)", 0.0, 100.0, 30.0)

    with col2:
        VIP = st.selectbox("VIP (True = 1, False = 0)", ["True", "False"])
        RoomService = st.number_input("RoomService", 0.0, step=1.0)
        FoodCourt = st.number_input("FoodCourt", 0.0, step=1.0)
        ShoppingMall = st.number_input("ShoppingMall", 0.0, step=1.0)
        Spa = st.number_input("Spa", 0.0, step=1.0)
        VRDeck = st.number_input("VRDeck", 0.0, step=1.0)

    submit = st.form_submit_button("Prediksi 🚀")

# -----------------------------
# RUN PREDIKSI
# -----------------------------
if submit:

    df = pd.DataFrame([{
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

    df = preprocess(df)

    try:
        pred = model.predict(df)[0]
        st.success(f"🚀 Hasil Prediksi: **{bool(pred)}**")
    except Exception as e:
        st.error("❌ Error pada prediksi. Periksa input.")
        st.code(str(e))
