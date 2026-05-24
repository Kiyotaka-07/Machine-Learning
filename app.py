import streamlit as st
import pandas as pd
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CO₂ Emission Predictor",
    page_icon="🚗",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# =========================
# VEHICLE CLASS MAPPING
# =========================
vehicle_weights = {
    'TWO-SEATER': 1100,
    'MINICOMPACT': 1150,
    'SUBCOMPACT': 1250,
    'COMPACT': 1350,
    'STATION WAGON - SMALL': 1400,
    'MID-SIZE': 1600,
    'STATION WAGON - MID-SIZE': 1650,
    'SUV - SMALL': 1800,
    'SUV - STANDARD': 2200,
    'MINIVAN': 2200,
    'PICKUP TRUCK - SMALL': 2300,
    'PICKUP TRUCK - STANDARD': 2600,
    'VAN - PASSENGER': 2500,
    'VAN - CARGO': 2700,
    'SPECIAL PURPOSE VEHICLE': 2800
}

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1f2937;
    text-align: center;
    margin-top: 20px;
}

.small-text {
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📘 About")

    st.write("""
    This application predicts vehicle
    **CO₂ emissions (g/km)** using a trained
    Machine Learning model.

    ### Models Used
    - XGBoost
    - Random Forest
    - Decision Tree
    - Linear Regression

    ### Input Features
    - Engine Size
    - Cylinders
    - Vehicle Class
    - Transmission
    - Fuel Type
    """)

# =========================
# TITLE
# =========================
st.title("🚗 CO₂ Emission Predictor")

st.write("""
Predict vehicle CO₂ emissions using a Machine Learning model trained on vehicle specifications.
""")

st.divider()

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    engine_size = st.slider(
        "Engine Size (L)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.1
    )

    cylinders = st.slider(
        "Cylinders",
        min_value=2,
        max_value=16,
        value=4,
        step=1
    )

with col2:
    vehicle_class = st.selectbox(
        "Vehicle Class",
        list(vehicle_weights.keys())
    )

    transmission = st.selectbox(
        "Transmission Type",
        ['A', 'AM', 'AS', 'AV', 'M']
    )

fuel = st.selectbox(
    "Fuel Type",
    ['D', 'E', 'N', 'X', 'Z']
)

vehicle_weight = vehicle_weights[vehicle_class]

st.caption(f"Estimated Vehicle Weight: {vehicle_weight} kg")

st.divider()

# =========================
# PREDICTION
# =========================
if st.button("Predict CO₂ Emission"):

    input_data = {
        'Engine Size(L)': engine_size,
        'Cylinders': cylinders,
        'vehicle_class_num': vehicle_weight,
        'trans_A': 0,
        'trans_AM': 0,
        'trans_AS': 0,
        'trans_AV': 0,
        'trans_M': 0,
        'fuel_D': 0,
        'fuel_E': 0,
        'fuel_N': 0,
        'fuel_X': 0,
        'fuel_Z': 0
    }

    input_data[f'trans_{transmission}'] = 1
    input_data[f'fuel_{fuel}'] = 1

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.markdown(
        f"""
        <div class="result-box">
            <h2>Estimated CO₂ Emission</h2>
            <h1>{prediction:.2f} g/km</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # emission category
    if prediction < 150:
        st.success("Low emission vehicle ✅")

    elif prediction < 250:
        st.warning("Moderate emission vehicle ⚠️")

    else:
        st.error("High emission vehicle 🚨")

# =========================
# FOOTER
# =========================
st.divider()

st.markdown(
    """
    <p class="small-text">
    Built with Streamlit and Machine Learning.
    </p>
    """,
    unsafe_allow_html=True
)