import streamlit as st
import pandas as pd
import joblib

from model_utils import predict_and_monitor,load_model
from Preprocessing_utils import (
    handle_nulls,
    binary_encoding,
    view_encoding,
    finishing_encoding
)

st.set_page_config(page_title="Real Estate Price Prediction", layout="centered")

model = load_model()

st.title("🏡 Real Estate Price Prediction App")

st.write("Fill in the property details below to estimate its market price.")

district = st.selectbox(
    "District",
    ["Fifth Settlement", "New Cairo", "Maadi", "Nasr City", "Heliopolis", "6 October", "Sheikh Zayed"]
)

compound_options = [
    "Katameya Plaza",
    "Katameya Dunes",
    "Madinaty B4",
    "Madinaty B1",
    "Madinaty B2",
    "Madinaty B3",
    "Mirage City",
    "Gardenia Springs",
    "Mountain View",
    "Hyde Park",
    "Palm Hills",
    "Moon Valley",
    "Rehab 3",
    "Rehab 2",
    "Rehab 1",
    "Rehab 4",
    "Katameya Heights",
    "Lake View",
    "No compound"
]

compound_name = st.selectbox("Compound Name", compound_options)

seller_type = st.selectbox("Seller Type", ["Owner", "Broker"])

finishing_type = st.selectbox(
    "Finishing Type",
    ["Unfinished", "Semi-finished", "Lux", "Super Lux"]
)

view_type = st.selectbox(
    "View Type",
    ["Street", "Garden", "Compound", "Nile"]
)

area_sqm = st.number_input("Area (sqm)", min_value=60, max_value=400, step=1)
bedrooms = st.number_input("Bedrooms", min_value=2, max_value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=2, step=1)

floor_number = st.number_input("Floor Number", min_value=1, max_value=20, step=1)
building_age_years = st.number_input("Building Age (Years)", min_value=1, max_value=50, step=1)

distance_to_auc_km = st.number_input("Distance to AUC (km)", min_value=0.0, max_value=50.0)
distance_to_mall_km = st.number_input("Distance to Mall (km)", min_value=0.0, max_value=50.0)
distance_to_metro_km = st.number_input("Distance to Metro (km)", min_value=0.0, max_value=50.0)

has_parking = st.selectbox("Parking", ["Yes", "No"])
has_security = st.selectbox("Security", ["Yes", "No"])
has_amenities = st.selectbox("Amenities", ["Yes", "No"])
has_balcony = st.selectbox("Balcony", ["Yes", "No"])
is_negotiable = st.selectbox("Negotiable Price", ["Yes", "No"])

if st.button("Predict Price"):

    df = pd.DataFrame([{
        "area_sqm": area_sqm,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floor_number": floor_number,
        "building_age_years": building_age_years,
        "district": district,
        "compound_name": compound_name,
        "distance_to_auc_km": distance_to_auc_km,
        "distance_to_mall_km": distance_to_mall_km,
        "distance_to_metro_km": distance_to_metro_km,
        "finishing_type": finishing_type,
        "has_balcony": has_balcony,
        "has_parking": has_parking,
        "has_security": has_security,
        "has_amenities": has_amenities,
        "view_type": view_type,
        "seller_type": seller_type,
        "is_negotiable": is_negotiable
    }], columns=[
        "area_sqm",
        "bedrooms",
        "bathrooms",
        "floor_number",
        "building_age_years",
        "district",
        "compound_name",
        "distance_to_auc_km",
        "distance_to_mall_km",
        "distance_to_metro_km",
        "finishing_type",
        "has_balcony",
        "has_parking",
        "has_security",
        "has_amenities",
        "view_type",
        "seller_type",
        "is_negotiable"
    ])

    df = handle_nulls(df)
    df = binary_encoding(df)
    df = view_encoding(df)
    df = finishing_encoding(df)

    prediction = predict_and_monitor(model, df)

    st.success(f"💰 Estimated Property Price: {prediction:,.0f} EGP")
