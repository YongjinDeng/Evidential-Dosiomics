# 07_SHAP_Interpretability.py (修复数值问题版)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import warnings
warnings.filterwarnings('ignore')

RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
FEATURES_CSV = os.path.join(RESULT_DIR, "Fractional_Evidential_Features.csv")
LABELS_CSV = os.path.join(RESULT_DIR, "clinical_labels_simple.csv")

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("\n" + "="*70)
    print("🧠 步骤07：SHAP 核心机制解释图 (修复数值版)")
    print("="*70)

    # 加载数据
    df_feat = pd.read_csv(FEATURES_CSV)
    df_label = pd.read_csv(LABELS_CSV)
    data = pd.merge(df_feat, df_label, on='Patient_ID')

    X = data.filter(regex='^(CLEAN_|BIOMARKER_|RISK_)')
    y = data['Label']
    
    # 特征筛选
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    selector = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, 
                                  class_weight='balanced', random_state=42)
    selector.fit(X_scaled_df, y)

    coefs = selector.coef_[0]
    non_zero_idx = np.where(coefs != 0)[0]
    selected_features = X.columns[non_zero_idx].tolist()
    
    print(f"🔥 冠军模型锁定了 {len(selected_features)} 个核心特征: {selected_features}")

    if len(selected_features) == 0:
        return
    
    # 重新拟合模型
    X_refit = X_scaled_df[selected_features]
    final_model = LogisticRegression(penalty=None, class_weight='balanced', random_state=42)
    final_model.fit(X_refit, y)

    # 计算 SHAP 值
    explainer = shap.LinearExplainer(final_model, X_refit)
    shap_values = explainer.shap_values(X_refit)
    
    # 打印调试信息
    print(f"\n📊 SHAP 值统计:")
    for i, feat in enumerate(selected_features):
        print(f"   {feat}:")
        print(f"      SHAP 均值: {shap_values[:, i].mean():.4f}")
        print(f"      |SHAP| 均值: {np.abs(shap_values[:, i]).mean():.4f}")
        print(f"      模型系数: {final_model.coef_[0][i]:.4f}")
    
    # 准备数据
    shap_importance = np.abs(shap_values).mean(0)
    sorted_idx = np.argsort(shap_importance)[::-1]
    
    # 特征名称简化
    feature_labels = []
    for feat in selected_features:
        if 'BIOMARKER_glcm_DifferenceVariance' in feat:
            feature_labels.append('Difference Variance\n(Biomarker)')
        elif 'CLEAN_glcm_ClusterTendency' in feat:
            feature_labels.append('Cluster Tendency\n(Clean)')
        else:
            feature_labels.append(feat.split('_')[-1])
    
    features_sorted = [feature_labels[i] for i in sorted_idx]
    shap_importance_sorted = shap_importance[sorted_idx]
    
    print(f"\n📊 排序后的重要性: {list(zip(features_sorted, shap_importance_sorted))}")
    
    # 创建图形：左右布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white', dpi=150)
    
    # ===== 左图：Bar Plot =====
    colors_bar = ['#1f77b4', '#ff7f0e']
    bars = ax1.barh(range(len(selected_features)), shap_importance_sorted, 
                    color=colors_bar[:len(selected_features)], edgecolor='black', 
                    linewidth=0.8, height=0.5)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, shap_importance_sorted)):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=15, fontweight='bold')
    
    ax1.set_yticks(range(len(selected_features)))
    ax1.set_yticklabels(features_sorted, fontsize=15)
    ax1.set_xlabel('Mean |SHAP Value|', fontsize=15, fontweight='bold')
    ax1.set_xlim(0, max(shap_importance_sorted) * 1.2)
    ax1.invert_yaxis()
    ax1.grid(axis='x', linestyle=':', alpha=0.5)
    ax1.set_title('A  Global Feature Contribution', fontsize=18, fontweight='bold', loc='left')
    
    # ===== 右图：Beeswarm Plot =====
    shap_values_sorted = shap_values[:, sorted_idx]
    feature_names_display = [f.replace('\n', ' ') for f in features_sorted]
    
    np.random.seed(42)
    for i, (feat_name, shap_vals) in enumerate(zip(feature_names_display, shap_values_sorted.T)):
        # y 位置（从上到下）
        y_pos = np.ones(len(shap_vals)) * (len(selected_features) - 1 - i)
        # 添加 jitter
        jitter = np.random.normal(0, 0.08, len(shap_vals))
        
        # 根据 SHAP 值正负着色
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in shap_vals]
        
        ax2.scatter(shap_vals, y_pos + jitter, c=colors, s=35, alpha=0.6, edgecolors='none')
    
    # 添加参考线
    ax2.axvline(x=0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)
    
    # 设置轴
    ax2.set_yticks(range(len(selected_features)))
    ax2.set_yticklabels(feature_names_display[::-1], fontsize=15)
    ax2.set_ylim(-0.5, len(selected_features) - 0.5)
    ax2.set_xlabel('SHAP value', fontsize=15, fontweight='bold')
    ax2.set_title('B  Feature Impact Direction', fontsize=18, fontweight='bold', loc='left')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#d62728', alpha=0.6, label='Higher risk (Positive)'),
                       Patch(facecolor='#1f77b4', alpha=0.6, label='Lower risk (Negative)')]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=13, frameon=True)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(RESULT_DIR, "Main_Figure4_SHAP.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✅ SHAP 合并图已保存: {output_path}")
    
    # 输出最终结果
    print("\n" + "="*50)
    print("🏆 核心证据分析:")
    print("="*50)
    for i, feat in enumerate(selected_features):
        importance = shap_importance[i]
        coef = final_model.coef_[0][i]
        direction = "Positive (risk ↑)" if coef > 0 else "Negative (risk ↓)"
        short_name = feat.replace('BIOMARKER_glcm_', '').replace('CLEAN_glcm_', '')
        print(f"   {short_name:<35} | Importance: {importance:.4f} | {direction}")
    
    print("="*70)

if __name__ == "__main__":
    main()