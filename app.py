import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration & Styling - Ye hamesha sabse pehle chalega taaki UI load ho sake
st.set_page_config(page_title="Bank Marketing Predictor", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:38px !important; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 30px; }
    .predict-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏦 Bank Marketing Term Deposit Predictor</div>", unsafe_allow_html=True)

# ==========================================
# SAFE DEPENDENCY CHECKING
# ==========================================
# Check karenge ki heavyweight libraries installed hain ya nahi, taaki app crash na ho
libs_installed = True
missing_libs = []

try:
    import tensorflow as tf
except ImportError:
    libs_installed = False
    missing_libs.append("tensorflow")

try:
    import joblib
except ImportError:
    libs_installed = False
    missing_libs.append("joblib")

# Artifacts status flags
artifacts_loaded = False
preprocessor, model, model_columns = None, None, None

# Agar libraries installed hain, tabhi unhe load karne ka try karenge
if libs_installed:
    @st.cache_resource
    def load_resources():
        # ML artifacts ko load karna[cite: 2]
        prep = joblib.load('preprocessing.pkl')
        mdl = tf.keras.models.load_model('model.keras') # ya model.h5[cite: 2]
        cols = joblib.load('column.pkl')
        return prep, mdl, cols

    try:
        preprocessor, model, model_columns = load_resources()
        artifacts_loaded = True
        st.success("✅ Deep Learning Model & Preprocessor aligned successfully!")
    except Exception as e:
        st.error(f"⚠️ App is running but Model files missing: {e}. Check if 'model.keras', 'preprocessing.pkl' & 'column.pkl' exist.")
        # print("a")
else:
    st.warning(f"⚠️ Running in UI-Only Mode. Missing local system dependencies: {', '.join(missing_libs)}. Please run 'pip install -r requirements.txt'")

# ==========================================
# 2. RESPONSIVE UI INPUTS (Hamesha dikhega)
# ==========================================
st.subheader("📋 Enter Customer Details")

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
# 3. BACKEND PREDICTION LOGIC (Safe Trigger)
# ==========================================
if st.button("🔮 Predict Term Deposit Conversion", type="primary", use_container_width=True):
    
    # Check 1: Kya system me libraries installed hain?
    if not libs_installed:
        st.error(f"❌ Prediction blocked: Local machine is missing packages ({', '.join(missing_libs)}). Run 'pip install tensorflow joblib' first.")
    
    # Check 2: Kya artifacts folders me hain?
    elif not artifacts_loaded:
        st.error("❌ Prediction blocked: ML models or pipeline files are missing from the folder path.")
        
    else:
        try:
            # Input data ka raw DataFrame[cite: 2]
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
            
            # column.pkl ke mutabik features ka exact sequence set[cite: 2]
            if isinstance(model_columns, (list, np.ndarray, pd.Index)):
                input_data = input_data.reindex(columns=model_columns)
            
            # UI Data transformation[cite: 2]
            input_processed = preprocessor.transform(input_data)
            
            # Model prediction[cite: 2]
            prediction_prob = model.predict(input_processed)[0][0]
            
            st.markdown("### 🎯 Prediction Result")
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
                
        except Exception as prediction_error:
            st.error(f"❌ Core processing error: {prediction_error}")