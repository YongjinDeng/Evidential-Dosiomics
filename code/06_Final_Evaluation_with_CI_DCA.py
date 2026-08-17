# 06_Final_Evaluation_with_CI_DCA.py
"""
步骤06升级版：包含 95% 置信区间 (Bootstrap) 与 临床决策曲线 (DCA)
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
FEATURES_CSV = os.path.join(RESULT_DIR, "Fractional_Evidential_Features.csv")
LABELS_CSV = os.path.join(RESULT_DIR, "clinical_labels_simple.csv")
OUTPUT_FIG_ROC = os.path.join(RESULT_DIR, "Main_Figure3_ROC_CI.png")
OUTPUT_FIG_DCA = os.path.join(RESULT_DIR, "Main_Figure_DCA.png")

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    N = len(y_true)
    for thresh in thresholds:
        tp = np.sum((y_prob >= thresh) & (y_true == 1))
        fp = np.sum((y_prob >= thresh) & (y_true == 0))
        if thresh == 1.0:
            net_benefits.append(0.0)
            continue
        nb = (tp / N) - (fp / N) * (thresh / (1 - thresh))
        net_benefits.append(nb)
    return net_benefits

def main():
    print("\n" + "="*70)
    print("🏆 升级版步骤06：严格医学评估 (95% CI + DCA 决策曲线)")
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

    # 2. 交叉验证
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

    # 3. Patient-level 极值池化
    data['pred_prob'] = all_probs
    patient_results = data.groupby('Patient_ID').agg({
        'Label': 'first',
        'pred_prob': 'max'
    }).reset_index()

    final_y = patient_results['Label'].values
    final_prob = patient_results['pred_prob'].values
    
    # 4. Bootstrap 1000次计算 95% CI
    print("⏳ 正在进行 1000 次 Bootstrap 抽样计算 95% 置信区间...")
    n_bootstraps = 1000
    bootstrapped_aucs = []
    bootstrapped_briers = []
    
    np.random.seed(42)
    for i in range(n_bootstraps):
        # resample data (with replacement)
        indices = resample(np.arange(len(final_prob)), replace=True)
        if len(np.unique(final_y[indices])) < 2:
            continue # 如果抽到的全是同一个标签则跳过
        auc_val = roc_auc_score(final_y[indices], final_prob[indices])
        brier_val = brier_score_loss(final_y[indices], final_prob[indices])
        bootstrapped_aucs.append(auc_val)
        bootstrapped_briers.append(brier_val)
        
    auc_median = np.median(bootstrapped_aucs)
    auc_lower, auc_upper = np.percentile(bootstrapped_aucs, [2.5, 97.5])
    
    brier_median = np.median(bootstrapped_briers)
    brier_lower, brier_upper = np.percentile(bootstrapped_briers, [2.5, 97.5])

    print("\n" + "-"*50)
    print(f"📊 最终医学统计指标报告：")
    print(f"   Patient-Level AUC: {auc_median:.3f} (95% CI: {auc_lower:.3f} - {auc_upper:.3f})")
    print(f"   Patient-Level Brier: {brier_median:.3f} (95% CI: {brier_lower:.3f} - {brier_upper:.3f})")
    print("-" * 50)

    # 5. 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(final_y, final_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#C0392B', lw=3, 
             label=f'Evidential Dosiomics\nAUC = {auc_median:.3f} ({auc_lower:.2f}-{auc_upper:.2f})')
    plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--', label='Random Guess')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#C0392B')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('Receiver Operating Characteristic', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=12, frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_ROC, dpi=300)
    plt.close()

    # 6. 绘制 DCA 曲线 (Decision Curve Analysis)
    thresholds = np.linspace(0.01, 0.99, 100)
    nb_model = calculate_net_benefit(final_y, final_prob, thresholds)
    
    # 全部干预的 Net Benefit: (总正样本率) - (总负样本率) * (pt / (1-pt))
    prev = np.mean(final_y)
    nb_all = [prev - (1 - prev) * (pt / (1 - pt)) for pt in thresholds]
    nb_none = [0] * len(thresholds)

    plt.figure(figsize=(7, 6))
    plt.plot(thresholds, nb_model, color='#E74C3C', lw=3, label='Evidential Dosiomics')
    plt.plot(thresholds, nb_all, color='#3498DB', lw=2, linestyle='--', label='Treat All')
    plt.plot(thresholds, nb_none, color='black', lw=2, linestyle='-', label='Treat None')
    
    plt.xlim([0.0, 0.8])
    plt.ylim([-0.1, 0.6]) # 根据实际数据微调
    plt.xlabel('Threshold Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Net Benefit', fontsize=14, fontweight='bold')
    plt.title('Decision Curve Analysis (DCA)', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="upper right", fontsize=12, frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DCA, dpi=300)
    plt.close()

    print(f"✅ ROC图与DCA图已生成并保存！")
    print("="*70)

if __name__ == "__main__":
    main()
