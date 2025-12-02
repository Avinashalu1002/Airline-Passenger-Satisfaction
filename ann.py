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
    color: #F9FAFB;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stSidebar"] > div {{
    background: rgba(15,23,42,0.92);
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 0 25px rgba(59,130,246,0.4);
}}

.main-card {{
    background: rgba(15,23,42,0.88);
    padding: 2.5rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 40px rgba(0,0,0,0.6);
    border: 1px solid rgba(148,163,184,0.6);
}}

.section-box {{
    background: rgba(15,23,42,0.9);
    padding: 1.2rem;
    border-radius: 15px;
    border: 1px solid rgba(148,163,184,0.45);
    box-shadow: 0 0 25px rgba(15,23,42,0.8);
}}

.title {{
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    color: #E0F2FE;
    text-shadow: 0 0 22px #38BDF8;
    margin-bottom: 0.3rem;
}}

.subtitle {{
    text-align: center;
    font-size: 1rem;
    color: #E5E7EB;
    margin-bottom: 1.5rem;
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
    background-size: 200% 200%;
    transition: all 0.18s ease-out;
}}

.stButton > button:hover {{
    transform: translateY(-1px) scale(1.03);
    background-position: 100% 0%;
    box-shadow: 0 0 28px rgba(129,140,248,0.9);
}}

.metric-card {{
    background: radial-gradient(circle at top, rgba(56,189,248,0.25), transparent 55%),
                rgba(15,23,42,0.95);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(129,140,248,0.6);
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ===========================================================
# CUSTOM TRANSFORMER – ROBUST VERSION
# ===========================================================
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.label_encoders_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.columns is None:
            self.columns = X.columns.tolist()

        enc_map = {}
        for col in self.columns:
            le = LabelEncoder()
            vals = X[col].astype(str).fillna("missing")
            le.fit(vals)
            enc_map[col] = le

        self.label_encoders_ = enc_map
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Try both attributes so it works with however it was saved
        enc_map = getattr(self, "label_encoders_", None)
        if enc_map is None:
            enc_map = getattr(self, "encoders", None)

        if enc_map is None:
            raise AttributeError(
                "LabelEncoderTransformer has neither 'label_encoders_' nor 'encoders'. "
                "This usually means the transformer was never fitted."
            )

        for col, le in enc_map.items():
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
    except Exception:
        expected_cols = None
    return pipeline, model, expected_cols

pipeline, model, expected_cols = load_artifacts()

# ===========================================================
# SIDEBAR
# ===========================================================
with st.sidebar:
    st.markdown("## ✈️ Airline Happiness App")
    st.write(
        "Predict whether a passenger is **Satisfied** 💙 or "
        "**Not Satisfied** 💔 based on their flight details and service ratings."
    )
    if expected_cols:
        st.markdown("---")
        st.caption("Model was trained on these input features:")
        st.write(expected_cols)

# ===========================================================
# MAIN LAYOUT
# ===========================================================
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>Airline Passenger Satisfaction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Enter the passenger details and service experience to predict satisfaction.</div>",
    unsafe_allow_html=True,
)

with st.form("airline_form"):
    col1, col2 = st.columns(2)

    # Left side – passenger & flight info
    with col1:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.slider("Age", 10, 85, 28)
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
        flight_distance = st.slider("Flight Distance", 50, 6000, 1200)
        d_delay = st.slider("Departure Delay in Minutes", 0, 600, 5)
        a_delay = st.slider("Arrival Delay in Minutes", 0, 600, 5)
        st.markdown("</div>", unsafe_allow_html=True)

    # Right side – service ratings
    with col2:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        wifi = st.slider("Inflight wifi service", 0, 5, 3)
        time_conv = st.slider("Departure/Arrival time convenient", 0, 5, 3)
        online_book = st.slider("Ease of Online booking", 0, 5, 3)
        gate_loc = st.slider("Gate location", 0, 5, 3)
        food = st.slider("Food and drink", 0, 5, 3)
        online_board = st.slider("Online boarding", 0, 5, 3)
        seat = st.slider("Seat comfort", 0, 5, 3)
        entertainment = st.slider("Inflight entertainment", 0, 5, 3)
        onboard = st.slider("On-board service", 0, 5, 3)
        leg_room = st.slider("Leg room service", 0, 5, 3)
        baggage = st.slider("Baggage handling", 0, 5, 3)
        checkin = st.slider("Checkin service", 0, 5, 3)
        inflight_service = st.slider("Inflight service", 0, 5, 3)
        clean = st.slider("Cleanliness", 0, 5, 3)
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Predict Satisfaction ✨")

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
        "Arrival Delay in Minutes": a_delay,
    }

    df = pd.DataFrame([data])

    # Align features to what pipeline was trained on
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

    try:
        transformed = pipeline.transform(df)
        prob = float(model.predict(transformed)[0][0])
        label = "Satisfied ✈️💙" if prob >= 0.5 else "Not Satisfied 💔"

        st.markdown("---")
        res1, res2 = st.columns([2, 1])

        with res1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.write(f"**Prediction:** {label}")
            st.markdown("</div>", unsafe_allow_html=True)

        with res2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Satisfaction Probability", f"{prob * 100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        st.progress(max(0.0, min(prob, 1.0)))

    except Exception as e:
        st.error("❌ Error while predicting.")
        st.code(str(e), language="text")

st.markdown("</div>", unsafe_allow_html=True)
