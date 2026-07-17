import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="BFA-SVM Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection Dashboard")
st.subheader("Hybrid Binary Firefly Algorithm & Support Vector Machine")

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to load artifacts safely
@st.cache_resource
def load_artifacts():
    try:
        clf_bfa = joblib.load('clf_bfa.joblib')
        scaler_time = joblib.load('scaler_time.joblib')
        scaler_amount = joblib.load('scaler_amount.joblib')
        best_mask = joblib.load('best_mask.joblib')
        res_base = joblib.load('res_base.joblib')
        res_bfa = joblib.load('res_bfa.joblib')
        feature_names = joblib.load('feature_names.joblib')
        
        df_hist = pd.read_csv('table_4_1_bfa_convergence.csv')
        df_feat = pd.read_csv('table_4_2_feature_selection.csv')
        df_comp = pd.read_csv('table_4_3_comparative_analysis.csv')
        
        return clf_bfa, scaler_time, scaler_amount, best_mask, res_base, res_bfa, feature_names, df_hist, df_feat, df_comp
    except Exception as e:
        return None, None, None, None, None, None, None, None, None, None

clf_bfa, scaler_time, scaler_amount, best_mask, res_base, res_bfa, feature_names, df_hist, df_feat, df_comp = load_artifacts()

if clf_bfa is None:
    st.error("Training artifacts not found! Please run the offline training script first (`train.py`).")
    st.stop()

st.write("---")
st.write("### 1. Single Transaction Prediction")
st.write("Input the transaction parameters below to predict if it is Fraudulent or Genuine.")

# Get only the features selected by the BFA
selected_features = [f for i, f in enumerate(feature_names) if best_mask[i] == 1]

# Create input form for the selected features only
with st.form("prediction_form"):
    st.write(f"##### Transaction Features ({len(selected_features)} Selected by BFA)")
    
    cols = st.columns(4)
    input_values = {}
    
    for i, feature in enumerate(selected_features):
        col = cols[i % 4]
        input_values[feature] = col.number_input(f"{feature}", value=0.0, format="%.4f")
        
    submitted = st.form_submit_button("Predict Fraud Status")
    
    if submitted:
        # Create a dataframe from the inputs
        df_new = pd.DataFrame([input_values])
        
        # Scale 'Time' and 'Amount' if they were selected by the BFA
        if 'Time' in selected_features:
            df_new['Time'] = scaler_time.transform(df_new['Time'].values.reshape(-1, 1))
        if 'Amount' in selected_features:
            df_new['Amount'] = scaler_amount.transform(df_new['Amount'].values.reshape(-1, 1))
        
        # Get raw numpy array for the selected features directly
        X_new_sub = df_new.values
        
        # Predict
        pred = clf_bfa.predict(X_new_sub)[0]
        
        st.write("#### Prediction Result:")
        if pred == 1:
            st.error("🚨 **FRAUDULENT TRANSACTION DETECTED** 🚨")
        else:
            st.success("✅ **GENUINE TRANSACTION** ✅")

st.write("---")
st.write("### 2. Training Benchmarks (Offline Results)")

st.write("The models were trained offline on the unbalanced Kaggle dataset using SMOTE strictly on the training partition. Here are the benchmarks evaluated on the untouched test set.")

def plot_confusion_matrix(cm, title, ax):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

# Display Confusion Matrices
st.write("#### Confusion Matrices (Test Set)")
fig_cm, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_confusion_matrix(res_base['Confusion Matrix'], "Baseline SVM (All Features)", axes[0])
plot_confusion_matrix(res_bfa['Confusion Matrix'], f"Hybrid BFA-SVM ({int(np.sum(best_mask))} Features)", axes[1])
st.pyplot(fig_cm)

# Display Performance Chart
st.write("#### Performance Comparison")
metrics = ['Accuracy', 'F1-Score', 'Recall', 'MCC', 'AUPRC']
df_metrics = df_comp.set_index('Model')[metrics].T

fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
df_metrics.plot(kind='bar', ax=ax_bar)
ax_bar.set_title("Performance Metrics")
ax_bar.set_ylabel("Score")
ax_bar.set_ylim(0, 1.1)
plt.xticks(rotation=45)
st.pyplot(fig_bar)

st.write("---")
st.write("### 3. Academic Deliverables (Chapter 4)")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Table 4.1: BFA Convergence**")
    st.dataframe(df_hist, height=200)
    st.download_button("Download Table 4.1", convert_df(df_hist), "table_4_1_bfa_convergence.csv", "text/csv", key="t41")
    
with col2:
    st.write("**Table 4.2: Feature Selection Matrix**")
    st.dataframe(df_feat, height=200)
    st.download_button("Download Table 4.2", convert_df(df_feat), "table_4_2_feature_selection.csv", "text/csv", key="t42")
    
with col3:
    st.write("**Table 4.3: Comparative Analysis**")
    st.dataframe(df_comp)
    st.download_button("Download Table 4.3", convert_df(df_comp), "table_4_3_comparative_analysis.csv", "text/csv", key="t43")
