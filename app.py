import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define function to make predictions
def predict_flood(input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title("Flood Prediction")

st.write("""
This application predicts the likelihood of a flood based on various environmental factors.
""")

# Define the input fields
monsoon_intensity = st.slider('Monsoon Intensity', 1, 10)
topography_drainage = st.slider('Topography Drainage', 1, 10)
river_management = st.slider('River Management', 1, 10)
deforestation = st.slider('Deforestation', 1, 10)
urbanization = st.slider('Urbanization', 1, 10)
climate_change = st.slider('Climate Change', 1, 10)
dams_quality = st.slider('Dams Quality', 1, 10)
siltation = st.slider('Siltation', 1, 10)
agricultural_practices = st.slider('Agricultural Practices', 1, 10)
encroachments = st.slider('Encroachments', 1, 10)
ineffective_disaster_preparedness = st.slider('Ineffective Disaster Preparedness', 1, 10)
drainage_systems = st.slider('Drainage Systems', 1, 10)
coastal_vulnerability = st.slider('Coastal Vulnerability', 1, 10)
landslides = st.slider('Landslides', 1, 10)
watersheds = st.slider('Watersheds', 1, 10)
deteriorating_infrastructure = st.slider('Deteriorating Infrastructure', 1, 10)
population_score = st.slider('Population Score', 1, 10)
wetland_loss = st.slider('Wetland Loss', 1, 10)
inadequate_planning = st.slider('Inadequate Planning', 1, 10)
political_factors = st.slider('Political Factors', 1, 10)

# Prepare input data for prediction
input_data = pd.DataFrame([[
    monsoon_intensity,
    topography_drainage,
    river_management,
    deforestation,
    urbanization,
    climate_change,
    dams_quality,
    siltation,
    agricultural_practices,
    encroachments,
    ineffective_disaster_preparedness,
    drainage_systems,
    coastal_vulnerability,
    landslides,
    watersheds,
    deteriorating_infrastructure,
    population_score,
    wetland_loss,
    inadequate_planning,
    political_factors
]], columns=[
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
    'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems', 'CoastalVulnerability',
    'Landslides', 'Watersheds', 'DeterioratingInfrastructure', 'PopulationScore',
    'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
])

# Display prediction result
if st.button('Predict'):
    result = predict_flood(input_data)
    if result:
        st.write("Prediction: **Flood likely**")
    else:
        st.write("Prediction: **Flood unlikely**")
