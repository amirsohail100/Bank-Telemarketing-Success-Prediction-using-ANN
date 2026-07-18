import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Page Configuration & Styling
st.set_page_config(page_title="Bank Marketing Predictor", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:38px !important; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 30px; }
    .predict-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏦 Bank Marketing Term Deposit Predictor</div>", unsafe_allow_html=True)

# ==========================================
# 1. LOAD ARTIFACTS (Model & Preprocessor)
# ==========================================
@st.cache_resource
def load_resources():
    # Aapka preprocessing.pkl aur model.keras/h5 yahan load hoga
    preprocessor = joblib.load('preprocessing.pkl')
    model = tf.keras.models.load_model('model.keras') # ya 'model.h5'
    return preprocessor, model

try:
    preprocessor, model = load_resources()
    st.success("✅ Model and Preprocessor loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}. Make sure 'model.keras' and 'preprocessing.pkl' are in the same folder.")
    st.stop()

# ==========================================
# 2. RESPONSIVE UI INPUTS (Form Columns)
# ==========================================
st.subheader("📋 Enter Customer Details")

# Form ko do split panels mein divide kiya responsive design ke liye
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🔢 Numerical Demographics")
    age = st.slider("Age", min_value=17, max_value=98, value=35)
    campaign = st.number_input("Number of Contacts during Campaign", min_value=1, max_value=50, value=1)
    previous = st.number_input("Number of Contacts before Campaign", min_value=0, max_value=10, value=0)
    
    st.markdown("### 📈 Economic Indicators")
    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.99)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.85)
    nr_employed = st.number_input("Number of Employees Indicator", value=5191.0)
    
    # Hamara custom binary column
    was_contacted_val = st.radio("Was the customer contacted previously?", ["No", "Yes"])
    was_contacted = 1 if was_contacted_val == "Yes" else 0

with col2:
    st.markdown("### 🏷️ Categorical Profile")
    job = st.selectbox("Job Type", ['housemaid', 'services', 'admin.', 'technician', 'blue-collar', 'unemployed', 'retired', 'entrepreneur', 'management', 'student', 'self-employed', 'unknown'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    education = st.selectbox("Education Level", ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'unknown', 'university.degree', 'illiterate'])
    default = st.selectbox("Has Credit in Default?", ['no', 'unknown', 'yes'])
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])
    
    st.markdown("### 📞 Contact History")
    contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
    month = st.selectbox("Last Contact Month", ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox("Last Contact Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    poutcome = st.selectbox("Outcome of Previous Marketing Campaign", ['nonexistent', 'failure', 'success'])

st.write("---")

# ==========================================
# 3. BACKEND PREDICTION LOGIC
# ==========================================
if st.button("🔮 Predict Term Deposit Conversion", type="primary", use_container_width=True):
    
    # 1. Input data ka raw DataFrame banao (Sare 19 columns sequential list format mein)
    input_data = pd.DataFrame([{
        'age': age, 'campaign': campaign, 'previous': previous, 
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 
        'nr.employed': nr_employed, 'was_contacted': was_contacted,
        'job': job, 'marital': marital, 'education': education, 
        'default': default, 'housing': housing, 'loan': loan, 
        'contact': contact, 'month': month, 'day_of_week': day_of_week, 
        'poutcome': poutcome
    }])
    
    # 2. UI Data ko usi pkl preprocessor se scale aur transform karo (Generates 62 columns)
    input_processed = preprocessor.transform(input_data)
    
    # 3. Model se probability predict karo
    prediction_prob = model.predict(input_processed)[0][0]
    
    # 4. Result Dashboard Output
    st.markdown("### 🎯 Prediction Result")
    
    # Convert to percentage
    confidence = prediction_prob * 100
    
    if prediction_prob >= 0.5:
        st.markdown(f"""
            <div class='predict-box' style='background-color: #D1FAE5; color: #065F46; border: 2px solid #34D399;'>
                🎉 SUCCESS: Customer is highly likely to subscribe! <br>
                <span style='font-size: 18px;'>Confidence Probability: {confidence:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='predict-box' style='background-color: #FEE2E2; color: #991B1B; border: 2px solid #F87171;'>
                ❌ NO CONVERSION: Customer will likely reject the deposit offer. <br>
                <span style='font-size: 18px;'>Confidence Probability: {100 - confidence:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)