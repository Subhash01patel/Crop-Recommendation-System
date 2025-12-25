import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------- Load saved model and encoders ----------
model_gbc = pickle.load(open("models/model_gbc.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
rainfall_encoder = pickle.load(open("models/rainfall_encoder.pkl", "rb"))
ph_encoder = pickle.load(open("models/ph_encoder.pkl", "rb"))

# ---------- Feature engineering (same as notebook) ----------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df['NPK'] = (df['N'] + df['P'] + df['K']) / 3
    df['THI'] = df['temperature'] * df['humidity'] / 100
    df['rainfall_level'] = pd.cut(df['rainfall'],
                                  bins=[0, 50, 100, 200, 300],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    def ph_category(p):
        if p < 5.5:
            return 'Acidic'
        elif p <= 7.5:
            return 'Neutral'
        else:
            return 'Alkaline'
    df['ph_category'] = df['ph'].apply(ph_category)
    df['temp_rain_interaction'] = df['temperature'] * df['rainfall']
    df['ph_rain_interaction'] = df['ph'] * df['rainfall']
    return df

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N','P','K','temperature','humidity','ph','rainfall'])
    input_df = feature_engineer(input_df)
    input_df['rainfall_level'] = rainfall_encoder.transform(input_df['rainfall_level'])
    input_df['ph_category'] = ph_encoder.transform(input_df['ph_category'])
    prediction_encoded = model_gbc.predict(input_df)
    return label_encoder.inverse_transform(prediction_encoded)[0]

# ---------- Streamlit UI ----------
st.title("🌱 Crop Recommendation System")

st.write("Enter soil and weather parameters to get a recommended crop.")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.number_input("Phosphorus (P)", 0, 200, 50)
    K = st.number_input("Potassium (K)", 0, 200, 50)
    ph = st.number_input("pH", 0.0, 14.0, 6.5, step=0.1)

with col2:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0, step=0.1)

if st.button("Recommend Crop"):
    result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    st.success(f"✅ Recommended Crop: **{result}**")
