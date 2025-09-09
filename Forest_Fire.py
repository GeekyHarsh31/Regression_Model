import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io

# Load model and scaler
with open('ridge.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Page config
st.set_page_config(page_title="FWI Predictor", page_icon="🔥", layout="wide")

# Hero section
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='color: #2e4d2e;'>🔥 Forest Weather Index Predictor</h1>
    <p style='font-size: 18px; max-width: 700px; margin: auto;'>
        Predict fire risk using real-time environmental data from Algerian forests. 
        Upload data or enter values manually to estimate the Fire Weather Index (FWI).
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar: Region + CSV Upload
st.sidebar.header("🌍 Region Selection")
region_map = {"Bejaia": 1, "Sidi-Bel Abbes": 2}
region_name = st.sidebar.selectbox("Choose Region", options=list(region_map.keys()))
region = region_map[region_name]

st.sidebar.header("📁 Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Manual input
st.subheader("📥 Environmental Parameters")
col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", value=25.0, help="Ambient temperature in Celsius")
    RH = st.number_input("Relative Humidity (%)", value=45.0, help="Humidity level in percentage")
    Ws = st.number_input("Wind Speed (km/h)", value=10.0, help="Wind speed in kilometers per hour")
    Rain = st.number_input("Rain (mm)", value=0.0, help="Rainfall amount in millimeters")

with col2:
    FFMC = st.number_input("FFMC Index", value=85.0, help="Fine Fuel Moisture Code")
    DMC = st.number_input("DMC Index", value=50.0, help="Duff Moisture Code")
    DC = st.number_input("DC Index", value=200.0, help="Drought Code")
    ISI = st.number_input("ISI Index", value=5.0, help="Initial Spread Index")

# Single prediction
if st.button("🔍 Predict FWI"):
    input_data = np.array([[region, temp, RH, Ws, Rain, FFMC, DMC, DC, ISI]])

    try:
        scaled_input = scaler.transform(input_data)
        predicted_fwi = model.predict(scaled_input)[0]

        st.success(f"Estimated FWI: **{predicted_fwi:.2f}**")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_fwi,
            title={'text': "Fire Weather Index"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Downloadable report
        if st.button("📥 Download Prediction Report"):
            report = f"""
            Forest Weather Index Prediction Report
            --------------------------------------
            Region: {region_name}
            Temperature: {temp} °C
            Humidity: {RH} %
            Wind Speed: {Ws} km/h
            Rain: {Rain} mm
            FFMC: {FFMC}
            DMC: {DMC}
            DC: {DC}
            ISI: {ISI}
            Predicted FWI: {predicted_fwi:.2f}
            """
            st.download_button("Download Report", data=report, file_name="FWI_Report.txt", mime="text/plain")

    except ValueError as e:
        st.error(f"Input mismatch: {e}")
        st.info("Make sure the feature order and count match your training setup.")

# Batch prediction
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        scaled_batch = scaler.transform(df)
        batch_predictions = model.predict(scaled_batch)

        df['Predicted FWI'] = batch_predictions
        st.subheader("✅ Batch Predictions")
        st.dataframe(df)

        st.download_button("Download Results as CSV", data=df.to_csv(index=False), file_name="fwi_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Historical trend (simulated)
history = pd.DataFrame({
    "Date": pd.date_range(end=pd.Timestamp.today(), periods=10),
    "FWI": np.random.uniform(10, 80, size=10)
})

st.subheader("📊 Historical FWI Trend")
fig = px.line(history, x="Date", y="FWI", markers=True, title="FWI Over Time")
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.9rem;'>Model trained on Algerian Forest Dataset | Built by Harsh 🌟</div>", unsafe_allow_html=True)
