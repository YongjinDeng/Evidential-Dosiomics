# 06_Final_ROC_Evaluation.py
"""
步骤05：最终患者级预测性能评估与 ROC 曲线绘制
- 算法：GroupKFold + Lasso (L1) 特征选择 + Worst-Case Pooling
- 对应 AUC 0.7412 的冠军模型
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, auc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ================== 配置 ==================
RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
FEATURES_CSV = os.path.join(RESULT_DIR, "Fractional_Evidential_Features.csv")
LABELS_CSV = os.path.join(RESULT_DIR, "clinical_labels_simple.csv")
OUTPUT_ROC = os.path.join(RESULT_DIR, "Main_Figure3_ROC.png")

# 设置顶刊绘图风格
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("\n" + "="*70)
    print("🏆 步骤05：绘制最终冠军模型 ROC 曲线 (Evidential Dosiomics)")
    print("="*70)

    # 1. 加载数据
    df_feat = pd.read_csv(FEATURES_CSV)
    df_label = pd.read_csv(LABELS_CSV)
    data = pd.merge(df_feat, df_label, on='Patient_ID')

    X = data.filter(regex='^(CLEAN_|BIOMARKER_|RISK_)')
    y = data['Label']
    groups = data['Patient_ID']
    
    gkf = GroupKFold(n_splits=5)
    all_probs = np.zeros(len(y))

    # 2. 交叉验证评估
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 核心算法：与 Table 1 保持绝对一致的 L1 Lasso
        model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, 
                                   class_weight='balanced', random_state=42)
        model.fit(X_train_s, y_train)
        all_probs[test_idx] = model.predict_proba(X_test_s)[:, 1]

    # 3. 极值池化 (Worst-Case Pooling)
    data['pred_prob'] = all_probs
    patient_results = data.groupby('Patient_ID').agg({
        'Label': 'first',
        'pred_prob': 'max'
    }).reset_index()

    final_y = patient_results['Label']
    final_prob = patient_results['pred_prob']
    
    final_auc = roc_auc_score(final_y, final_prob)
    brier = brier_score_loss(final_y, final_prob)

    print(f"🔥 Final Patient-Level AUC: {final_auc:.4f}")
    print(f"🔥 Final Brier Score: {brier:.4f}")

    # 4. 绘制高颜值 ROC 曲线 (Figure 3)
    fpr, tpr, _ = roc_curve(final_y, final_prob)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#C0392B', lw=3, 
             label=f'Evidential Dosiomics (AUC = {final_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--', label='Random Guess')
    
    # 填充曲线下面积
    plt.fill_between(fpr, tpr, alpha=0.1, color='#C0392B')

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title('Receiver Operating Characteristic (Patient-Level)', fontsize=15, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=12, frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_ROC, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✅ 冠军模型 ROC 曲线已保存至: {OUTPUT_ROC}")
    print("="*70)

if __name__ == "__main__":
    main()