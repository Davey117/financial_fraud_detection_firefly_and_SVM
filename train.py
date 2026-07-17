import time
import joblib
import pandas as pd
import numpy as np
from data_processor import load_and_preprocess_data, split_and_balance_data
from bfa_optimizer import BinaryFireflyOptimizer
from evaluator import train_and_evaluate, generate_outputs

def main():
    print("Loading data...")
    X, y, scaler_time, scaler_amount = load_and_preprocess_data('creditcard_10percent.csv')
    feature_names = X.columns.tolist()
    
    print("Splitting and balancing data...")
    X_train, y_train, X_test, y_test = split_and_balance_data(X, y)
    
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    y_test_np = y_test.values
    
    print("Running BFA Optimization...")
    optimizer = BinaryFireflyOptimizer(N=10, max_iter=15, alpha=0.2, gamma=1.0)
    best_mask, bfa_history = optimizer.optimize(X_train_np, y_train_np)
    
    print(f"Selected {int(np.sum(best_mask))} features. Training Baseline SVM on full 80% train set...")
    res_base, clf_base = train_and_evaluate(
        X_train_np, y_train_np, X_test_np, y_test_np, 
        feature_mask=None, model_name="Baseline SVM"
    )
    
    print("Training Hybrid BFA-SVM on full 80% train set...")
    res_bfa, clf_bfa = train_and_evaluate(
        X_train_np, y_train_np, X_test_np, y_test_np, 
        feature_mask=best_mask, model_name="Hybrid BFA-SVM"
    )
    
    print("Saving outputs and models...")
    generate_outputs(bfa_history, best_mask, feature_names, res_base, res_bfa)
    
    # Save objects for inference
    joblib.dump(scaler_time, 'scaler_time.joblib')
    joblib.dump(scaler_amount, 'scaler_amount.joblib')
    joblib.dump(best_mask, 'best_mask.joblib')
    joblib.dump(clf_bfa, 'clf_bfa.joblib')
    joblib.dump(clf_base, 'clf_base.joblib')
    
    # Save the training results to show in Streamlit
    joblib.dump(res_base, 'res_base.joblib')
    joblib.dump(res_bfa, 'res_bfa.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    
    print("Training complete! Artifacts saved.")

if __name__ == "__main__":
    main()
