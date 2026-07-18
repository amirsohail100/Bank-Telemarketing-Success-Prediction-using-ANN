# Bank Telemarketing Success Prediction Using ANN

[![Streamlit App](https://static.streamlit.io/badge/sticker/blue.svg)](YOUR_WORKING_LINK_HERE)

## 🎯 Project Description

A production-ready deep learning pipeline predicting bank telemarketing success via an ANN model trained on the UCI dataset with 82% accuracy. Features a dynamic Streamlit UI and FastAPI backend optimized with Pydantic validation to enforce strict input data boundaries and prevent inefficient model resource token burn. By processing complex client profiles alongside crucial macroeconomic indicators, the architecture replaces raw scripts with a resource-efficient inference application.

## 📊 Model Accuracy & Performance

- **Model Framework:** TensorFlow / Keras (Artificial Neural Network - ANN)[cite: 2]
- **Target Objective:** Binary Classification (Predicting `yes` / `no` for term deposit subscription)[cite: 1]
- **Validation Accuracy:** **82%** 🔥
- **Key Insight:** The integration of 5 social and economic contextual attributes (such as employment variation rate and consumer price index) significantly enhances predictive performance compared to using traditional demographic features alone[cite: 1].

## 🧠 Neural Network Architecture

The deep learning architecture is constructed with the following layers:

1. **Input Layer:** Designed to receive preprocessed feature vectors derived from the 20 core dataset attributes[cite: 1].
2. **Hidden Layers:** Multiple Dense layers utilizing `ReLU` activation, combined with Dropout regularization to mitigate overfitting during training.
3. **Output Layer:** A single continuous Dense neuron configured with a `Sigmoid` activation function to compute precise probability scores between 0 and 1.

## 🛠️ Data Preprocessing Pipeline

To prevent data leakage and ensure 100% feature consistency during live inference, the pipeline utilizes a robust preprocessing setup:

- **Categorical Encoding:** Multi-class categorical variables (e.g., job type, education, marital status) are transformed using `OneHotEncoder`[cite: 1].
- **Numerical Normalization:** Continuous numeric attributes (e.g., age, consumer confidence index, euribor 3 month rate) are scaled using `StandardScaler`[cite: 1].
- **Pipeline Synchronization:** Transformations are bundled into an SKLearn `ColumnTransformer`, loaded via `preprocessing.pkl` and strictly aligned with live inputs using `column.pkl`[cite: 2].

## 🌐 Live Demo & Interface

- **Deployment URL:** [Launch Live Streamlit Application](YOUR_WORKING_LINK_HERE)

### Application User Interface

_(Replace the placeholder image below with your actual UI screenshot path after uploading it to your repository)_
![Streamlit App Interface](YOUR_IMAGE_PATH_OR_URL_HERE)

## 📚 Citation & Dataset Credits

This predictive framework is built upon the public dataset and research provided by:

> _Moro, S., Cortez, P., and Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems._[cite: 1]

End-to-end Deep Learning application using an ANN model to predict term deposit subscription (82% accuracy). Features an interactive Streamlit frontend and FastAPI backend with strict Pydantic validation schemas.
