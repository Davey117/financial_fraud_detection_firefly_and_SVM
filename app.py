import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set Page Config
st.set_page_config(page_title="BFA-SVM Fraud Detection Dashboard", layout="wide")

# Custom CSS for rich aesthetics and dynamic card animations
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Font styling */
        html, body, [class*="css"], .stMarkdown, p, div, label {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Gradient Header Panel */
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
            text-align: center;
        }
        .main-header h1 {
            color: white !important;
            margin: 0;
            font-weight: 700;
            font-size: 2.6rem;
            letter-spacing: -0.5px;
        }
        .main-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.2rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        /* Prediction Result Cards */
        .genuine-card {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 6px solid #28a745;
            padding: 1.5rem;
            border-radius: 8px;
            color: #155724;
            box-shadow: 0 4px 6px rgba(40, 167, 69, 0.1);
            margin-top: 1.5rem;
        }
        .genuine-card h4 {
            color: #155724 !important;
            margin-top: 0;
            font-weight: 600;
        }
        
        .fraud-card {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left: 6px solid #dc3545;
            padding: 1.5rem;
            border-radius: 8px;
            color: #721c24;
            box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
            margin-top: 1.5rem;
            animation: pulse 2s infinite;
        }
        .fraud-card h4 {
            color: #721c24 !important;
            margin-top: 0;
            font-weight: 600;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
        }
        
        /* Metric block styling */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 1.2rem;
            border-radius: 10px;
            box-shadow: 0 3px 6px rgba(0,0,0,0.02);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.06);
            border-color: var(--primary-color);
        }
        
        /* Sidebar Specifications Box */
        .sidebar-spec {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 1.2rem;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.2);
            margin-top: 1.5rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.02);
        }
        .sidebar-spec strong {
            color: var(--text-color);
            display: block;
            margin-bottom: 0.5rem;
        }
        .sidebar-spec code {
            background-color: rgba(128, 128, 128, 0.15);
            color: var(--primary-color);
            padding: 0.15rem 0.3rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
        }
        
        /* Tab formatting */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

# Helper function to convert dataframe to CSV bytes
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

# Load Artifacts
clf_bfa, scaler_time, scaler_amount, best_mask, res_base, res_bfa, feature_names, df_hist, df_feat, df_comp = load_artifacts()

if clf_bfa is None:
    st.error("Training artifacts not found! Please run the offline training script first (`train.py`).")
    st.stop()

# Inject Custom CSS styles
inject_custom_css()

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>Credit Card Fraud Detection Dashboard</h1>
    <p>Hybrid Binary Firefly Algorithm (BFA) & Support Vector Machine (SVM) Classification</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation Control
st.sidebar.markdown("### 🛡️ Dashboard Control")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard Overview",
        "🔮 Prediction Playground",
        "📊 Model Evaluation & Benchmarks",
        "🧬 BFA Convergence & Features",
        "📂 Academic Deliverables"
    ]
)

# Sidebar specs card
st.sidebar.markdown("""
<div class="sidebar-spec">
    <strong>🔧 Engine Specifications:</strong>
    • Base Classifier: <code>SVM (RBF Kernel)</code><br>
    • Feature Optimization: <code>Binary Firefly (BFA)</code><br>
    • Balanced Technique: <code>SMOTE (Train set only)</code><br>
    • Dimension Reduction: <code>30 ➔ 16 features</code><br>
    • Feature Selection Rate: <code>53.3%</code>
</div>
""", unsafe_allow_html=True)

# Selected features list
selected_features = [f for i, f in enumerate(feature_names) if best_mask[i] == 1]

# ----------------- PAGE 1: DASHBOARD OVERVIEW -----------------
if page == "🏠 Dashboard Overview":
    st.write("### 🏠 Dashboard Overview & Key Metrics")
    st.write("Compare the baseline SVM model (trained on all 30 features) versus the hybrid BFA-SVM model (trained on 16 optimized features chosen via Binary Firefly Optimization).")

    # KPI Metrics block at the top
    st.markdown("#### High-Level Performance Comparison")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall Performance", "80.00%", "0.00% (Maintained)", delta_color="off")
    col2.metric("Features Selected", "16 / 30", "-46.67% Reduction", delta_color="normal")
    col3.metric("Hybrid SVM Accuracy", "98.60%", "-0.42%", delta_color="inverse")
    col4.metric("Hybrid SVM MCC", "0.270", "-0.049", delta_color="inverse")
    
    st.write("---")
    st.markdown("#### Two-Tab System Explanation")
    
    # 2-Tab Explanation requested by User
    tab_concept, tab_workflow = st.tabs(["💡 Hybrid BFA-SVM Concept", "⚙️ Pipeline & Optimization Workflow"])
    
    with tab_concept:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.markdown("""
            ##### The Challenge of Fraud Detection
            Credit card transaction datasets are heavily imbalanced. Typically, less than **0.2%** of transactions are fraudulent. 
            Standard Machine Learning models trained on such datasets skew heavily towards predicting the majority class (Genuine), resulting in poor Recall.
            
            ##### The Hybrid Solution
            To build a robust fraud detection engine, this dashboard implements a multi-stage hybrid algorithm:
            1. **Class Balancing (SMOTE)**: Synthetic Minority Over-sampling Technique is applied strictly to the training partition to resolve class imbalance without data leakage.
            2. **Feature Optimization (Binary Firefly)**: Feature selection is modeled as a binary optimization problem. A swarm of fireflies searches the feature space to discover the sub-features that maximize classification power.
            3. **Support Vector Machine (SVM)**: A Radial Basis Function (RBF) kernel SVM is trained on the selected feature subsets, providing high-efficiency predictions.
            """)
        with col_c2:
            st.info("""
            **💡 Why Feature Selection?**
            Reducing the number of features from 30 to 16:
            *   Speeds up prediction response times.
            *   Reduces storage and throughput requirements.
            *   Eliminates noisy or redundant transaction variables.
            """)
            
    with tab_workflow:
        st.markdown("""
        ##### Optimization & Inference Pipeline
        The step-by-step process of the BFA-SVM pipeline is outlined below:
        """)
        
        flow_cols = st.columns(4)
        with flow_cols[0]:
            st.markdown("""
            **Step 1: Scaling**
            Robust scaling is applied to `Time` and `Amount` using median and IQR to handle extreme transaction outliers.
            """)
        with flow_cols[1]:
            st.markdown("""
            **Step 2: SMOTE Over-sampling**
            Sythetic fraud instances are created on the training set to prevent SVM bias towards genuine transactions.
            """)
        with flow_cols[2]:
            st.markdown("""
            **Step 3: BFA Optimization**
            BFA simulates firefly attraction based on light intensity (fitness score = MCC value) to find optimal feature mask.
            """)
        with flow_cols[3]:
            st.markdown("""
            **Step 4: Inference**
            New transactions are scaled, masked to 16 features, and evaluated by the optimized SVM model.
            """)

# ----------------- PAGE 2: PREDICTION PLAYGROUND -----------------
elif page == "🔮 Prediction Playground":
    st.write("### 🔮 Interactive Prediction Playground")
    st.write("Adjust transaction features below or choose from one of the presets to test classification behavior.")
    
    # Preset selectbox
    preset_choice = st.selectbox(
        "Select Transaction Preset Profile:",
        ["Custom (Zero-initialized)", "Genuine Transaction Preset", "Fraudulent Transaction Preset"]
    )
    
    # Preset Values Mapping
    preset_values = {}
    if preset_choice == "Genuine Transaction Preset":
        preset_values = {
            'Time': 80000.0,
            'Amount': 15.0
        }
        for f in selected_features:
            if f not in ['Time', 'Amount']:
                preset_values[f] = 0.0
    elif preset_choice == "Fraudulent Transaction Preset":
        # Values discovered via offline classifier inspection
        preset_values = {
            'Time': -79804.42, 'V4': 5.09, 'V6': -1.00, 'V8': -1.21, 'V9': -2.87, 
            'V13': 1.27, 'V14': -8.19, 'V15': -3.20, 'V19': 0.07, 'V21': 4.24, 
            'V22': -0.24, 'V25': 1.36, 'V26': -3.19, 'V27': 1.28, 'V28': -0.56, 
            'Amount': 230.04
        }
    else:
        for f in selected_features:
            preset_values[f] = 0.0
            
    # Form layout
    with st.form("prediction_form"):
        st.markdown(f"##### Feature Inputs ({len(selected_features)} Features Selected by BFA)")
        
        cols = st.columns(4)
        input_values = {}
        
        for i, feature in enumerate(selected_features):
            col = cols[i % 4]
            default_val = float(preset_values.get(feature, 0.0))
            input_values[feature] = col.number_input(f"{feature}", value=default_val, format="%.4f")
            
        submitted = st.form_submit_button("Predict Status")
        
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
            
            # Display beautifully styled result card
            if pred == 1:
                st.markdown("""
                <div class="fraud-card">
                    <h4>🚨 FRAUDULENT TRANSACTION DETECTED</h4>
                    <p>The system classifies this transaction as <strong>SUSPICIOUS/FRAUDULENT</strong>. A high risk pattern has been identified.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="genuine-card">
                    <h4>✅ GENUINE TRANSACTION</h4>
                    <p>The system classifies this transaction as <strong>NORMAL/GENUINE</strong>. No suspicious fraud pattern matches were found.</p>
                </div>
                """, unsafe_allow_html=True)

# ----------------- PAGE 3: MODEL EVALUATION & BENCHMARKS -----------------
elif page == "📊 Model Evaluation & Benchmarks":
    st.write("### 📊 Performance & Comparative Analysis")
    st.write("View the detailed performance benchmarks generated by running baseline SVM vs optimized Hybrid BFA-SVM models on the untouched test partition.")
    
    tab_cm, tab_charts = st.tabs(["📝 Confusion Matrices", "📈 Performance Comparisons"])
    
    with tab_cm:
        st.write("#### Confusion Matrices Comparison (Test Set)")
        
        def plot_confusion_matrix(cm, title, ax):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                        annot_kws={"size": 14, "weight": "bold"})
            ax.set_title(title, fontsize=14, pad=12, weight="bold")
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_xticklabels(['Genuine', 'Fraud'], fontsize=10)
            ax.set_yticklabels(['Genuine', 'Fraud'], fontsize=10)
            
        fig_cm, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_confusion_matrix(res_base['Confusion Matrix'], "Baseline SVM (All 30 Features)", axes[0])
        plot_confusion_matrix(res_bfa['Confusion Matrix'], f"Hybrid BFA-SVM (16 Selected Features)", axes[1])
        plt.tight_layout()
        st.pyplot(fig_cm)
        st.caption("Confusion matrices demonstrate that the BFA-SVM retains high true positive detection capability while drastically reducing feature dimensionality.")
        
    with tab_charts:
        st.write("#### Performance Comparison Bar Chart")
        metrics = ['Accuracy', 'F1-Score', 'Recall', 'MCC', 'AUPRC']
        df_metrics = df_comp.set_index('Model')[metrics].T
        
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        df_metrics.plot(kind='bar', ax=ax_bar, color=['#2a5298', '#28a745'])
        ax_bar.set_title("Performance Comparison Across Key Metrics", fontsize=14, pad=12, weight="bold")
        ax_bar.set_ylabel("Score Value", fontsize=12)
        ax_bar.set_xlabel("Evaluation Metrics", fontsize=12)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_bar)
        
        # Display side-by-side table
        st.write("#### Comparative Benchmark Table")
        st.dataframe(df_comp, width='stretch')

# ----------------- PAGE 4: BFA CONVERGENCE & FEATURES -----------------
elif page == "🧬 BFA Convergence & Features":
    st.write("### 🧬 Binary Firefly Algorithm (BFA) Convergence & Selection Status")
    st.write("Analyze the optimization history and feature selection state generated by the wrapper algorithm.")
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.write("#### BFA Convergence History")
        fig_conv, ax_conv = plt.subplots(figsize=(7, 5))
        ax_conv.plot(df_hist['Iteration'], df_hist['Best MCC'], marker='o', color='#1e3c72', linewidth=2.5, markersize=6, label='Best MCC')
        ax_conv.set_title("BFA Best Fitness (MCC) Convergence Profile", fontsize=12, weight="bold")
        ax_conv.set_xlabel("Algorithm Iterations", fontsize=10)
        ax_conv.set_ylabel("Best Matthews Correlation Coefficient (MCC)", fontsize=10)
        ax_conv.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_conv)
        
    with col_plot2:
        st.write("#### Feature Selection Status Matrix")
        # Custom visual representation of selected vs unselected features
        df_feat_sorted = df_feat.copy()
        df_feat_sorted['Selected Color'] = df_feat_sorted['Selected [Yes/No]'].map({'Yes': '#28a745', 'No': '#dc3545'})
        
        fig_feat, ax_feat = plt.subplots(figsize=(7, 7))
        y_pos = np.arange(len(df_feat_sorted))
        
        # Plot horizontal bars
        ax_feat.barh(y_pos, [1] * len(df_feat_sorted), color=df_feat_sorted['Selected Color'].values, height=0.6)
        ax_feat.set_yticks(y_pos)
        ax_feat.set_yticklabels(df_feat_sorted['Feature Name'], fontsize=9)
        ax_feat.invert_yaxis()
        ax_feat.set_title("Feature Mask Selection State (Green=Selected, Red=Discarded)", fontsize=12, weight="bold")
        ax_feat.set_xlabel("Selection Mask State", fontsize=10)
        ax_feat.set_xticks([])
        plt.tight_layout()
        st.pyplot(fig_feat)

# ----------------- PAGE 5: ACADEMIC DELIVERABLES -----------------
elif page == "📂 Academic Deliverables":
    st.write("### 📂 Academic Deliverables (Chapter 4 CSV Results)")
    st.write("Download or preview raw experimental datasets generated during the model configuration phase.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Table 4.1: BFA Convergence**")
        st.dataframe(df_hist, height=350, width='stretch')
        st.download_button(
            label="📥 Download Table 4.1", 
            data=convert_df(df_hist), 
            file_name="table_4_1_bfa_convergence.csv", 
            mime="text/csv", 
            key="t41_new"
        )
        
    with col2:
        st.markdown("**Table 4.2: Feature Selection Matrix**")
        st.dataframe(df_feat, height=350, width='stretch')
        st.download_button(
            label="📥 Download Table 4.2", 
            data=convert_df(df_feat), 
            file_name="table_4_2_feature_selection.csv", 
            mime="text/csv", 
            key="t42_new"
        )
        
    with col3:
        st.markdown("**Table 4.3: Comparative Analysis**")
        st.dataframe(df_comp, height=350, width='stretch')
        st.download_button(
            label="📥 Download Table 4.3", 
            data=convert_df(df_comp), 
            file_name="table_4_3_comparative_analysis.csv", 
            mime="text/csv", 
            key="t43_new"
        )
