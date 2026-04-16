# 05_Ablation_Study.py
"""
消融实验对比表 - 精简版
- 只输出一个合并的消融结果表格
- 不生成多余的中间文件
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
FEATURES_CSV = os.path.join(RESULT_DIR, "Fractional_Evidential_Features.csv")
GLOBAL_FEATURES_CSV = os.path.join(RESULT_DIR, "Fractional_Global_Features.csv")
LABELS_CSV = os.path.join(RESULT_DIR, "clinical_labels_simple.csv")
OUTPUT_FILE = os.path.join(RESULT_DIR, "Ablation_Results.csv")  # 只保留一个输出文件

def evaluate_strategy(X, y, groups):
    """统一的评估函数"""
    gkf = GroupKFold(n_splits=5)
    all_probs = np.zeros(len(y))
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, 
                                   class_weight='balanced', random_state=42)
        model.fit(X_train_s, y_train)
        all_probs[test_idx] = model.predict_proba(X_test_s)[:, 1]
        
    temp_df = pd.DataFrame({'Patient_ID': groups, 'Label': y, 'Prob': all_probs})
    patient_res = temp_df.groupby('Patient_ID').agg({'Label': 'first', 'Prob': 'max'})
    
    auc = roc_auc_score(patient_res['Label'], patient_res['Prob'])
    brier = brier_score_loss(patient_res['Label'], patient_res['Prob'])
    return auc, brier

def main():
    print("\n" + "="*70)
    print("📊 消融实验 (Ablation Study)")
    print("="*70)
    
    df_label = pd.read_csv(LABELS_CSV)
    df_feat = pd.read_csv(FEATURES_CSV)
    data = pd.merge(df_feat, df_label, on='Patient_ID')
    
    y = data['Label']
    groups = data['Patient_ID']
    
    clean_cols = [c for c in data.columns if c.startswith('CLEAN_')]
    bio_cols = [c for c in data.columns if c.startswith('BIOMARKER_')]
    risk_cols = [c for c in data.columns if c.startswith('RISK_')]
    
    results = []
    
    # 1. 传统剂量组学
    print("   📉 Traditional Dosiomics...")
    auc, brier = evaluate_strategy(data[clean_cols], y, groups)
    results.append({'Method': 'Traditional Dosiomics', 'ROI': 'Penumbra', 'AUC': auc, 'Brier': brier})
    
    # 2. 解耦组学（半影区）
    print("   📈 Decoupled Dosiomics (Penumbra)...")
    auc, brier = evaluate_strategy(data[clean_cols + bio_cols], y, groups)
    results.append({'Method': 'Decoupled Dosiomics', 'ROI': 'Penumbra', 'AUC': auc, 'Brier': brier})
    
    # 3. 全器官消融（如果存在）
    if os.path.exists(GLOBAL_FEATURES_CSV):
        print("   🌍 Global ROI (Ablation)...")
        df_global = pd.read_csv(GLOBAL_FEATURES_CSV)
        data_global = pd.merge(df_global, df_label, on='Patient_ID')
        clean_cols_g = [c for c in data_global.columns if c.startswith('CLEAN_')]
        bio_cols_g = [c for c in data_global.columns if c.startswith('BIOMARKER_')]
        
        if len(clean_cols_g) + len(bio_cols_g) > 0:
            auc, brier = evaluate_strategy(data_global[clean_cols_g + bio_cols_g], y, groups)
            results.append({'Method': 'Decoupled Dosiomics', 'ROI': 'Global (Whole Organ)', 'AUC': auc, 'Brier': brier})
    
    # 输出结果表格
    print("\n" + "="*70)
    print("📊 消融实验结果汇总")
    print("="*70)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # 只保存一个文件
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 结果已保存: {OUTPUT_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()