import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path, scaler_time=None, scaler_amount=None):
    """
    Load the credit card dataset and apply RobustScaler to 'Time' and 'Amount'.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please upload it via the dashboard.")
    
    # Check if expected columns are present
    if 'Time' not in df.columns or 'Amount' not in df.columns or 'Class' not in df.columns:
        raise ValueError("Dataset must contain 'Time', 'Amount', and 'Class' columns.")
        
    if scaler_time is None:
        scaler_time = RobustScaler()
        df['Time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    else:
        df['Time'] = scaler_time.transform(df['Time'].values.reshape(-1, 1))
        
    if scaler_amount is None:
        scaler_amount = RobustScaler()
        df['Amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    else:
        df['Amount'] = scaler_amount.transform(df['Amount'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y, scaler_time, scaler_amount

def split_and_balance_data(X, y):
    """
    Perform 80/20 train-test split with stratification.
    Apply SMOTE strictly to the TRAINING set to prevent data leakage.
    Returns: X_train_smote, y_train_smote, X_test, y_test
    """
    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    return X_train_smote, y_train_smote, X_test, y_test
