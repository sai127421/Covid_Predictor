import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="COVID-19 Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI
st.markdown("""
    <style>
    .main {
        background-color: #f4f7f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .header-box h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
    }
    .header-box p {
        color: #e0e8f5;
        font-size: 1.05rem;
        margin: 0;
    }
    .section-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border-top: 4px solid #2a5298;
    }
    .section-title {
        color: #1e3c72;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    .result-box-high {
        background-color: #ffe6e6;
        border-left: 6px solid #d93025;
        padding: 1.5rem;
        border-radius: 8px;
        color: #8a1c14;
        margin-top: 1.5rem;
    }
    .result-box-low {
        background-color: #e6f4ea;
        border-left: 6px solid #1e8e3e;
        padding: 1.5rem;
        border-radius: 8px;
        color: #137333;
        margin-top: 1.5rem;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(42, 82, 152, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Load model dictionary (Model, OHE, LE)
@st.cache_resource
def load_artifacts(filepath="covid_model_prod.pkl"):
    return joblib.load(filepath)

try:
    artifacts = load_artifacts("covid_model_prod.pkl")
    model = artifacts["Model"]
    ohe = artifacts["OHE"]
    le = artifacts["LE"]
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# Header Banner
st.markdown("""
    <div class="header-box">
        <h1>🩺 COVID-19 Symptom & Exposure Screener</h1>
        <p>Enter symptoms, medical history, and recent exposure details to run the predictive model.</p>
    </div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Could not load `covid_model_prod.pkl`. Make sure it is in the same folder as `app.py`. Error: {load_error}")
    st.stop()

# Exact feature names matching OneHotEncoder (including trailing spaces)
FEATURE_GROUPS = {
    "Primary Symptoms": [
        ("Breathing Problem", "Breathing Problem"),
        ("Fever", "Fever"),
        ("Dry Cough", "Dry Cough"),
        ("Sore throat", "Sore Throat"),
        ("Running Nose", "Running Nose"),
        ("Headache", "Headache"),
        ("Fatigue ", "Fatigue"),
        ("Gastrointestinal ", "Gastrointestinal Issues"),
    ],
    "Pre-existing Medical Conditions": [
        ("Asthma", "Asthma"),
        ("Chronic Lung Disease", "Chronic Lung Disease"),
        ("Heart Disease", "Heart Disease"),
        ("Diabetes", "Diabetes"),
        ("Hyper Tension", "Hypertension"),
    ],
    "Exposure & Precautionary Habits": [
        ("Abroad travel", "Recent Abroad Travel"),
        ("Contact with COVID Patient", "Contact with COVID Patient"),
        ("Attended Large Gathering", "Attended Large Gathering"),
        ("Visited Public Exposed Places", "Visited Public Exposed Places"),
        ("Family working in Public Exposed Places", "Family Working in Public Exposed Places"),
        ("Wearing Masks", "Wearing Masks Regularly"),
        ("Sanitization from Market", "Sanitization of Market Items"),
    ]
}

user_inputs = {}

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for idx, (group_title, features) in enumerate(FEATURE_GROUPS.items()):
        with columns[idx]:
            st.markdown(f'<div class="section-card"><div class="section-title">{group_title}</div>', unsafe_allow_html=True)
            for raw_col, display_label in features:
                user_inputs[raw_col] = st.radio(
                    label=display_label,
                    options=["No", "Yes"],
                    horizontal=True,
                    key=raw_col
                )
            st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("Analyze Risk")

if submitted:
    ordered_cols = list(ohe.feature_names_in_)
    input_df = pd.DataFrame([[user_inputs[col] for col in ordered_cols]], columns=ordered_cols)

    encoded_features = ohe.transform(input_df)
    prediction_encoded = model.predict(encoded_features)
    prediction_label = le.inverse_transform(prediction_encoded)[0]
    probabilities = model.predict_proba(encoded_features)[0]

    class_names = le.classes_
    prob_dict = dict(zip(class_names, probabilities))
    confidence = prob_dict[prediction_label] * 100

    if prediction_label == "Yes":
        st.markdown(f"""
            <div class="result-box-high">
                <h3>⚠️ High Risk Detected (Prediction: Yes)</h3>
                <p><strong>Model Confidence:</strong> {confidence:.1f}%</p>
                <p>Based on the entered symptoms and exposure factors, the model indicates a positive risk profile. Please consult a healthcare professional.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box-low">
                <h3>✅ Low Risk Detected (Prediction: No)</h3>
                <p><strong>Model Confidence:</strong> {confidence:.1f}%</p>
                <p>Based on the entered parameters, the model indicates a negative risk profile. Continue following standard health precautions.</p>
            </div>
        """, unsafe_allow_html=True)