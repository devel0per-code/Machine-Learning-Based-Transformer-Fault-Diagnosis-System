import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

from src.generate_data import generate_synthetic_dga
from src.duval_triangle import plot_duval_triangle, duval_1_classification
from src.data_processing import load_and_preprocess_dataframe, prepare_training_data, prepare_explicit_train_test_data
from src.ml_models import train_models, evaluate_models
from src.anomaly_detection import train_isolation_forest, predict_anomalies

st.set_page_config(page_title="Transformer Fault Diagnosis", layout="wide", page_icon="⚡")

st.title("⚡ Machine Learning-Based Transformer Fault Diagnosis System")
st.markdown("Diagnose and predict transformer faults using DGA and the Duval Triangle Method.")

@st.cache_data
def get_synthetic_data():
    if not os.path.exists('synthetic_dga_data.csv'):
        df = generate_synthetic_dga(2000)
        df.to_csv('synthetic_dga_data.csv', index=False)
    else:
        df = pd.read_csv('synthetic_dga_data.csv')
    return df

# --- Sidebar for Data Management ---
st.sidebar.header("📁 Data Management")
st.sidebar.markdown("Upload your own datasets (you can select multiple files or drag and drop a folder) to train and evaluate the models. If empty, the system uses the default synthetic dataset.")

train_files = st.sidebar.file_uploader("Upload Training Data (CSV files or Folder)", type=["csv"], accept_multiple_files=True)
test_files = st.sidebar.file_uploader("Upload Testing Data (CSV files or Folder) [Optional]", type=["csv"], accept_multiple_files=True)

# Load appropriate data
if train_files:
    dfs_train = [pd.read_csv(f) for f in train_files]
    df_train = pd.concat(dfs_train, ignore_index=True)
    st.sidebar.success(f"Loaded Training Data: {len(df_train)} samples from {len(train_files)} files")
else:
    df_train = get_synthetic_data()

if test_files:
    dfs_test = [pd.read_csv(f) for f in test_files]
    df_test = pd.concat(dfs_test, ignore_index=True)
    st.sidebar.success(f"Loaded Testing Data: {len(df_test)} samples from {len(test_files)} files")
else:
    df_test = None

# --- Dataset Validation and Mapping ---
REQUIRED_COLUMNS = ['H2', 'CH4', 'C2H4', 'C2H2', 'CO', 'CO2', 'Fault_Type']

def handle_dataset(df, dataset_name):
    if df is None:
        return None
        
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if not missing_cols:
        return df
        
    st.warning(f"⚠️ {dataset_name} dataset has different columns. Please map your data below to proceed:")
    
    mapped_df = pd.DataFrame()
    cols = st.columns(len(REQUIRED_COLUMNS))
    for i, req_col in enumerate(REQUIRED_COLUMNS):
        with cols[i]:
            guess_idx = 0
            for j, c in enumerate(df.columns):
                if c.lower() == req_col.lower() or req_col.lower() in c.lower():
                    guess_idx = j
                    break
            
            selected_col = st.selectbox(f"'{req_col}'", options=list(df.columns), index=guess_idx, key=f"map_{dataset_name}_{req_col}")
            
            # Map column and coerce to numeric if it's a gas concentration
            mapped_df[req_col] = df[selected_col]
            if req_col != 'Fault_Type':
                mapped_df[req_col] = pd.to_numeric(mapped_df[req_col], errors='coerce').fillna(0)
                
    return mapped_df

df_train = handle_dataset(df_train, "Training")
df_test = handle_dataset(df_test, "Testing")

tabs = st.tabs(["📊 Data Overview", "🔺 Duval Triangle", "🤖 ML Diagnostics", "⚠️ Anomaly Detection"])

with tabs[0]:
    st.header("Training Dataset Overview")
    st.dataframe(df_train.head())
    st.markdown(f"**Total Training Samples:** {len(df_train)}")
    
    fig = px.histogram(df_train, x="Fault_Type", title="Fault Type Distribution (Training Data)", color="Fault_Type")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.header("Duval Triangle Analysis")
    st.markdown("Visualizing the training dataset based on %CH4, %C2H4, and %C2H2.")
    
    fig_duval = plot_duval_triangle(df_train)
    st.plotly_chart(fig_duval, use_container_width=True)
    
    st.subheader("Manual Data Entry for Duval Triangle")
    col1, col2, col3 = st.columns(3)
    with col1:
        ch4_in = st.number_input("CH4 (ppm)", value=100.0)
    with col2:
        c2h4_in = st.number_input("C2H4 (ppm)", value=50.0)
    with col3:
        c2h2_in = st.number_input("C2H2 (ppm)", value=10.0)
        
    if st.button("Diagnose via Duval 1"):
        fault = duval_1_classification(ch4_in, c2h4_in, c2h2_in)
        st.success(f"Duval Triangle 1 Classification: **{fault}**")

with tabs[2]:
    st.header("Machine Learning Diagnostics")
    
    with st.spinner("Preparing Data and Training Models..."):
        X_train_raw, y_train_raw, features = load_and_preprocess_dataframe(df_train)
        
        if df_test is not None:
            X_test_raw, y_test_raw, _ = load_and_preprocess_dataframe(df_test)
            X_train, X_test, y_train, y_test, scaler = prepare_explicit_train_test_data(
                X_train_raw, y_train_raw, X_test_raw, y_test_raw
            )
            eval_mode = "Explicit Test Set Provided"
        else:
            X_train, X_test, y_train, y_test, scaler = prepare_training_data(X_train_raw, y_train_raw)
            eval_mode = "Automatic 80/20 Split from Training Set"
            
        trained_models = train_models(X_train, y_train)
        results = evaluate_models(trained_models, X_test, y_test)
        
    st.info(f"**Evaluation Methodology:** {eval_mode}")
    st.subheader("Model Accuracies on Test Data")
    res_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    fig_acc = px.bar(res_df, x='Model', y='Accuracy', title="Model Accuracy Comparison", color='Model')
    st.plotly_chart(fig_acc, use_container_width=True)
    
    st.subheader("Predict from New Sample")
    cols = st.columns(6)
    new_data = {}
    gas_defaults = {'H2': 100.0, 'CH4': 50.0, 'C2H4': 20.0, 'C2H2': 5.0, 'CO': 200.0, 'CO2': 1500.0}
    for idx, (gas, default) in enumerate(gas_defaults.items()):
        with cols[idx]:
            new_data[gas] = st.number_input(f"{gas} ppm", value=default, key=f"ml_{gas}")
            
    if st.button("Run ML Prediction"):
        r_ch4_h2 = new_data['CH4'] / max(new_data['H2'], 1e-6)
        r_c2h2_c2h4 = new_data['C2H2'] / max(new_data['C2H4'], 1e-6)
        r_c2h2_ch4 = new_data['C2H2'] / max(new_data['CH4'], 1e-6)
        r_co2_co = new_data['CO2'] / max(new_data['CO'], 1e-6)
        
        input_features = [
            new_data['H2'], new_data['CH4'], new_data['C2H4'], new_data['C2H2'], 
            new_data['CO'], new_data['CO2'],
            r_ch4_h2, r_c2h2_c2h4, r_c2h2_ch4, r_co2_co
        ]
        
        input_scaled = scaler.transform([input_features])
        
        rf_model = trained_models['Random Forest']
        rf_pred = rf_model.predict(input_scaled)[0]
        
        xgb_model, le = trained_models['XGBoost']
        xgb_pred_encoded = xgb_model.predict(input_scaled)[0]
        xgb_pred = le.inverse_transform([xgb_pred_encoded])[0]
        
        st.write(f"**Random Forest Prediction:** {rf_pred}")
        st.write(f"**XGBoost Prediction:** {xgb_pred}")

with tabs[3]:
    st.header("Anomaly Detection (Early Warning System)")
    st.markdown("Using Isolation Forest to detect anomalous gas combinations that might indicate emerging faults.")
    
    # Train Isolation Forest on the training data
    iso_model = train_isolation_forest(X_train)
    st.success("Isolation Forest Model Trained on Training Set Baseline.")
    
    st.markdown("Let's analyze the current Training Dataset for anomalies:")
    
    X_train_raw_scaled = scaler.transform(X_train_raw)
    anomalies = predict_anomalies(iso_model, X_train_raw_scaled)
    df_anomalies = df_train.copy()
    df_anomalies['Anomaly'] = ['Anomaly' if x == -1 else 'Normal' for x in anomalies]
    
    fig_anom = px.scatter(df_anomalies, x='H2', y='CH4', color='Anomaly', title="Anomalies detected in H2 vs CH4 space", 
                          color_discrete_map={"Normal": "green", "Anomaly": "red"})
    st.plotly_chart(fig_anom, use_container_width=True)
