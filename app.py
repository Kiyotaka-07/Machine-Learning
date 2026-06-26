import streamlit as st
import pandas as pd
import pickle

# Page
st.set_page_config(
    page_title="CO₂ Emission Predictor",
    page_icon="🚗",
    layout="centered"
)

# Load
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Weight map
vehicle_weights = {
    "TWO-SEATER": 1100,
    "MINICOMPACT": 1150,
    "SUBCOMPACT": 1250,
    "COMPACT": 1350,
    "STATION WAGON - SMALL": 1400,
    "MID-SIZE": 1600,
    "STATION WAGON - MID-SIZE": 1650,
    "SUV - SMALL": 1800,
    "SUV - STANDARD": 2200,
    "MINIVAN": 2200,
    "PICKUP TRUCK - SMALL": 2300,
    "PICKUP TRUCK - STANDARD": 2600,
    "VAN - PASSENGER": 2500,
    "VAN - CARGO": 2700,
    "SPECIAL PURPOSE VEHICLE": 2800
}

# Display -> model value
transmission_options = {
    "Automatic (A)": "A",
    "Automated Manual (AM)": "AM",
    "Automatic Select Shift (AS)": "AS",
    "Continuously Variable (AV)": "AV",
    "Manual (M)": "M"
}

fuel_options = {
    "Diesel (D)": "D",
    "Ethanol (E)": "E",
    "Natural Gas (N)": "N",
    "Regular Gasoline (X)": "X",
    "Premium Gasoline (Z)": "Z"
}

# Style
st.markdown("""
<style>

.main {
    padding-top:1rem;
}

.stButton>button{
    width:100%;
    height:3em;
    border:none;
    border-radius:10px;
    background:#F24629;
    color:white;
    font-size:18px;
    font-weight:bold;
    transition:.3s;
}

.stButton>button:hover{
    background:#F0ABA3;
    transform:scale(1.02);
}

.stButton>button:active{
    transform:scale(.98);
}

.result-box{
    padding:20px;
    border-radius:12px;
    background:#1f2937;
    text-align:center;
    margin-top:20px;
}

.small-text{
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:

    st.header("📘 About")

    st.write("""
This app predicts **vehicle CO₂ emissions (g/km)** using a trained Machine Learning (XGBoost) model.

### Inputs
- Engine Size
- Cylinders
- Vehicle Class
- Transmission
- Fuel Type
""")

# Title
st.markdown("""
<h1 style='text-align:center'>
🚗 CO₂ Emission Predictor
</h1>

<p style='text-align:center;font-size:18px'>
Predict vehicle CO₂ emissions using a trained Machine Learning (XGBoost) model.
</p>
""", unsafe_allow_html=True)

st.divider()

# Inputs
col1, col2 = st.columns(2)

with col1:

    engine_size = st.slider(
        "Engine Size (L)",
        0.5,
        10.0,
        2.0,
        0.1
    )

    cylinders = st.slider(
        "Cylinders",
        2,
        16,
        4
    )

with col2:

    vehicle_class = st.selectbox(
        "Vehicle Class",
        list(vehicle_weights.keys())
    )

    selected_trans = st.selectbox(
        "Transmission Type",
        list(transmission_options.keys())
    )

selected_fuel = st.selectbox(
    "Fuel Type",
    list(fuel_options.keys())
)

vehicle_weight = vehicle_weights[vehicle_class]
transmission = transmission_options[selected_trans]
fuel = fuel_options[selected_fuel]

st.caption(f"Estimated Vehicle Weight: {vehicle_weight} kg")

st.divider()

# Predict
if st.button("Predict CO₂ Emission"):

    input_data = {
        "Engine Size(L)": engine_size,
        "Cylinders": cylinders,
        "vehicle_class_num": vehicle_weight
    }

    # One-hot
    for t in ["A", "AM", "AS", "AV", "M"]:
        input_data[f"trans_{t}"] = 0

    for f in ["D", "E", "N", "X", "Z"]:
        input_data[f"fuel_{f}"] = 0

    input_data[f"trans_{transmission}"] = 1
    input_data[f"fuel_{fuel}"] = 1

    # DataFrame
    input_df = pd.DataFrame([input_data])

    # Match training columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    st.markdown(f"""
<div class="result-box">
<h2>Estimated CO₂ Emission</h2>
<h1>{prediction:.2f} g/km</h1>
</div>
""", unsafe_allow_html=True)

    if prediction < 150:
        st.success("Low emission vehicle ✅")

    elif prediction < 250:
        st.warning("Moderate emission vehicle ⚠️")

    else:
        st.error("High emission vehicle 🚨")

st.divider()
