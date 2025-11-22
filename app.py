import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# App title
st.title('🏠 House Price Prediction App')
st.write('Predict California house prices using machine learning')

# Sidebar for inputs
st.sidebar.header('Input Features')

MedInc = st.sidebar.slider('Median Income (in $10,000s)', 0.5, 15.0, 3.0, 0.1)
HouseAge = st.sidebar.slider('House Age (years)', 1, 52, 20)
AveRooms = st.sidebar.slider('Average Rooms', 1.0, 15.0, 5.0, 0.1)
AveBedrms = st.sidebar.slider('Average Bedrooms', 1.0, 5.0, 2.0, 0.1)
Population = st.sidebar.slider('Population', 3, 10000, 1000, 10)
AveOccup = st.sidebar.slider('Average Occupancy', 1.0, 10.0, 3.0, 0.1)
Latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 37.0, 0.1)
Longitude = st.sidebar.slider('Longitude', -125.0, -114.0, -120.0, 0.1)

# Create input array
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, 
                        Population, AveOccup, Latitude, Longitude]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction button
if st.button('🔮 Predict Price'):
    prediction = model.predict(input_scaled)[0]
    st.success(f'### Predicted House Price: ${prediction * 100000:,.2f}')
    
    # Display input summary
    st.subheader('📊 Input Summary')
    input_df = pd.DataFrame(input_data, columns=[
        'Median Income', 'House Age', 'Avg Rooms', 
        'Avg Bedrooms', 'Population', 'Avg Occupancy', 
        'Latitude', 'Longitude'
    ])
    st.dataframe(input_df, use_container_width=True)
    
    # Additional insights
    st.subheader('💡 Insights')
    st.write(f"- Income level: {'High' if MedInc > 5 else 'Moderate' if MedInc > 2.5 else 'Low'}")
    st.write(f"- Property age: {'New' if HouseAge < 10 else 'Moderate' if HouseAge < 30 else 'Old'}")
    st.write(f"- Location: Lat {Latitude:.2f}, Long {Longitude:.2f}")