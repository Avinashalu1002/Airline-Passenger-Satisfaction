import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# ===========================================================
# PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="✈️ Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide"
)

# ===========================================================
# BEAUTIFUL BACKGROUND + PREMIUM UI
# ===========================================================
BACKGROUND_IMAGE = "https://images.pexels.com/photos/912050/pexels-photo-912050.jpeg"

custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

* {{
    font-family: 'Poppins', sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(
        rgba(0,0,0,0.55),
        rgba(0,0,0,0.75)
    ), url('{BACKGROUND_IMAGE}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.main-card {{
    background: rgba(255,255,255,0.12);
    padding: 2.5rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 40px rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.3);
}}

.section-box {{
    background: rgba(0,0,0,0.35);
    padding: 1.2rem;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 0 25px rgba(255,255,255,0.12);
}}

.title {{
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    color: #E0F2FE;
    text-shadow: 0 0 22px #38BDF8;
}}

.subtitle {{
    text-align: center;
    font-size: 1rem;
    color: #E0E0E0;
}}

.stButton > button {{
    width: 100%;
    border-radius: 50px;
    padding: 0.8rem;
    font-size: 1rem;
    font-weight: 600;
    background: linear-gradient(90deg, #38BDF8, #6366F1, #EC4899);
    color: white;
    border: none;
    box-shadow: 0 0 18px rgba(147,51,234,0.7);
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ===========================================================
# CUSTOM TRANSFORMER (needed for loading pipeline)
# ===========================================================
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.columns is None:
            self.columns = X.columns.tolist()

        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str).fillna("missing"))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for col, le in self.encoders.items():
            vals = X[col].astype(str).fillna("missing")
            known = set(le.classes_)
            X[col] = [le.transform([v])[0] if v in known else -1 for v in vals]
        return X

# ===========================================================
# LOAD PIPELINE + MODEL
# ===========================================================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("airline_pipeline_fitteds.pkl")
    model = tf.keras.models.load_model("Airline_Passenger_Satisfaction_ann.h5")

    try:
        expected_cols = list(pipeline.feature_names_in_)
    except:
        expected_cols = None

    return pipeline, model, expected_cols

pipeline, model, expected_cols = load_artifacts()

# ===========================================================
# UI LAYOUT
# ===========================================================
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>✈️ Airline Passenger Satisfaction Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter the passenger details and service ratings to predict satisfaction</div>", unsafe_allow_html=True)

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.slider("Age", 10, 85)
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
        flight_distance = st.slider("Flight Distance", 50, 6000)
        d_delay = st.slider("Departure Delay (minutes)", 0, 600)
        a_delay = st.slider("Arrival Delay (minutes)", 0, 600)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        wifi = st.slider("Inflight wifi service", 0, 5)
        time_conv = st.slider("Departure/Arrival time convenient", 0, 5)
        online_book = st.slider("Ease of Online booking", 0, 5)
        gate_loc = st.slider("Gate location", 0, 5)
        food = st.slider("Food and drink", 0, 5)
        online_board = st.slider("Online boarding", 0, 5)
        seat = st.slider("Seat comfort", 0, 5)
        entertainment = st.slider("Inflight entertainment", 0, 5)
        onboard = st.slider("On-board service", 0, 5)
        leg_room = st.slider("Leg room service", 0, 5)
        baggage = st.slider("Baggage handling", 0, 5)
        checkin = st.slider("Checkin service", 0, 5)
        inflight_service = st.slider("Inflight service", 0, 5)
        clean = st.slider("Cleanliness", 0, 5)
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Predict Satisfaction")

# ===========================================================
# PREDICTION
# ===========================================================
if submitted:
    data = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": travel_type,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": time_conv,
        "Ease of Online booking": online_book,
        "Gate location": gate_loc,
        "Food and drink": food,
        "Online boarding": online_board,
        "Seat comfort": seat,
        "Inflight entertainment": entertainment,
        "On-board service": onboard,
        "Leg room service": leg_room,
        "Baggage handling": baggage,
        "Checkin service": checkin,
        "Inflight service": inflight_service,
        "Cleanliness": clean,
        "Departure Delay in Minutes": d_delay,
        "Arrival Delay in Minutes": a_delay
    }

    df = pd.DataFrame([data])

    # Align columns with pipeline expectation
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

    try:
        transformed = pipeline.transform(df)
        prob = float(model.predict(transformed)[0][0])

        label = "Satisfied ✈️💙" if prob >= 0.5 else "Not Satisfied 💔"

        st.success(f"Prediction: **{label}**")
        st.metric("Satisfaction Probability", f"{prob * 100:.1f}%")

    except Exception as e:
        st.error("❌ Error while predicting.")
        st.code(str(e))
