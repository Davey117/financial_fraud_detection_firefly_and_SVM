import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef, average_precision_score, confusion_matrix

def train_and_evaluate(X_train, y_train, X_test, y_test, feature_mask=None, model_name="Baseline SVM"):
    """
    Train SVM and evaluate on test set.
    If feature_mask is provided, only uses selected features.
    """
    if feature_mask is not None:
        X_train_sub = X_train[:, feature_mask == 1]
        X_test_sub = X_test[:, feature_mask == 1]
        num_features = int(np.sum(feature_mask))
    else:
        X_train_sub = X_train
        X_test_sub = X_test
        num_features = X_train.shape[1]
        
    clf = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)
    
    start_time = time.time()
    clf.fit(X_train_sub, y_train)
    training_time = time.time() - start_time
    
    # Predict
    preds = clf.predict(X_test_sub)
    
    # SVM with RBF doesn't natively predict_proba without probability=True which is very slow. 
    # For AUPRC, we use decision_function instead.
    decision_scores = clf.decision_function(X_test_sub)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    auprc = average_precision_score(y_test, decision_scores)
    cm = confusion_matrix(y_test, preds)
    
    results = {
        'Model': model_name,
        'Selected Features Count': num_features,
        'Accuracy': acc,
        'F1-Score': f1,
        'Recall': recall,
        'MCC': mcc,
        'AUPRC': auprc,
        'Training Time': training_time,
        'Confusion Matrix': cm
    }
    
    return results, clf

def generate_outputs(bfa_history, best_mask, feature_names, results_baseline, results_bfa):
    """
    Generate DataFrames for Table 4.1, Table 4.2, and Table 4.3 and save as CSV.
    """
    # Table 4.1: BFA Convergence History
    df_history = pd.DataFrame(bfa_history)
    df_history.to_csv('table_4_1_bfa_convergence.csv', index=False)
    
    # Table 4.2: Feature Selection Matrix
    df_features = pd.DataFrame({
        'Feature Name': feature_names,
        'Selected [Yes/No]': ['Yes' if bool(m) else 'No' for m in best_mask]
    })
    df_features.to_csv('table_4_2_feature_selection.csv', index=False)
    
    # Table 4.3: Comparative Performance Analysis
    # Exclude confusion matrix from the comparative table
    res_base = {k: v for k, v in results_baseline.items() if k != 'Confusion Matrix'}
    res_bfa = {k: v for k, v in results_bfa.items() if k != 'Confusion Matrix'}
    
    df_comparison = pd.DataFrame([res_base, res_bfa])
    df_comparison.to_csv('table_4_3_comparative_analysis.csv', index=False)
    
    return df_history, df_features, df_comparison
