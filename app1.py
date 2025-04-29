import streamlit as st
import pandas as pd
import joblib
import datetime
import pytz
from pysolar.solar import get_altitude

# Load trained XGBoost model and scaler
model = joblib.load("solar1_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set location details
LATITUDE = 21.1458  # Nagpur, Maharashtra
LONGITUDE = 79.0882
TIMEZONE = "Asia/Kolkata"


# Function to compute solar elevation based on hour
def compute_solar_elevation(hour):
    # Assume a fixed date for simplicity
    date = datetime.datetime.utcnow().date()
    timestamp = datetime.datetime.combine(date, datetime.time(hour=hour, minute=0)).replace(tzinfo=pytz.utc)
    altitude = get_altitude(LATITUDE, LONGITUDE, timestamp)  # Solar Elevation Angle
    return altitude

# Streamlit App Interface
st.title("☀️ **ML-Based Solar Angle Prediction**")
st.markdown("""
    This app uses machine learning to predict the solar elevation angle based on input parameters.
    You can enter the power (in watts) and the hour of the day to get the predicted solar elevation angle.
""")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    power = st.sidebar.number_input("Power (Watts)", min_value=0.0, value=100.0, step=1.0)
    hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
    
    # Compute Solar Elevation
    solar_elevation = compute_solar_elevation(hour)
    
    # Return input data and solar elevation
    data = {
        "Power": [power],
        "Hour": [hour],
    }
    return pd.DataFrame(data), solar_elevation

# Get user inputs
df_input, solar_elevation = user_input_features()

# Scale input data for prediction
df_input_scaled = scaler.transform(df_input)

# Display the input values
st.subheader("Input Parameters")
st.write(f"Power (Watts): {df_input['Power'][0]}")
st.write(f"Hour of the Day: {df_input['Hour'][0]}")

st.markdown("---")
st.header("🌞 **Predict Solar Elevation**")

if st.button("Predict Solar Elevation"):
    st.success(f"🌞 **Predicted Solar Elevation**: {solar_elevation:.2f} degrees")
    st.markdown("##### Prediction details:")
    st.markdown("The solar elevation is based on the location and time provided. This prediction is made using a trained ML model.")
    

st.markdown("---")
st.markdown("""
    Made with ❤️ by Dil Raj.
""")
