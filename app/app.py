import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ========= SETTINGS =========
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
# EXACT training order:
FEATURE_ORDER = ['year', 'fuel', 'seller_type', 'transmission', 'engine', 'max_power']
LOG_TRAINED = True  # trained on log

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Predictor")
st.caption("Enter car details to estimate its price. All fields are required.")

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    year        = c1.number_input("Year", min_value=1985, max_value=2030, step=1, value=2017)
    engine_cc   = c2.number_input("Engine (CC)", min_value=10, max_value=10000, step=1, value=1248)

    c3, c4 = st.columns(2)
    max_power   = c3.number_input("Max Power (bhp)", min_value=1, max_value=2000, step=1, value=75)
    transmission= c4.selectbox("Transmission", ["Manual", "Automatic"])

    c5, c6 = st.columns(2)
    seller_type = c5.selectbox("Seller Type", ["Individual","Dealer","Trustmark Dealer"])
    fuel        = c6.selectbox("Fuel", ["Petrol","Diesel"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    # categorical mappings 
    fuel_map   = {"Petrol": 1, "Diesel": 2}
    trans_map  = {"Manual": 1, "Automatic": 2}
    seller_map = {"Individual": 1, "Dealer": 2, "Trustmark Dealer": 3}

    row = {
        'year': int(year),
        'fuel': fuel_map[fuel],
        'seller_type': seller_map[seller_type],
        'transmission': trans_map[transmission],
        'engine': float(engine_cc),
        'max_power': float(max_power),
    }
    X = pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

    # Load & predict
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model: {type(e).__name__}: {e}")
    else:
        try:
            yhat = float(model.predict(X)[0])
            price = float(np.exp(yhat)) if LOG_TRAINED else yhat
            st.success(f"Estimated Price: {price:,.0f}")
        except Exception as e:
            st.error(f"Prediction failed — {type(e).__name__}: {e}")
            with st.expander("Debug input"):
                st.write(X)
