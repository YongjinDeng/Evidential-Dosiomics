# 02_Univariate_RISK_Analysis.py
"""
步骤02：RISK特征单变量分析
展示RISK特征在高/低风险组间的差异，无需机器学习
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ================== 配置 ==================
RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
FEATURES_CSV = os.path.join(RESULT_DIR, "Fractional_Evidential_Features.csv")
LABELS_CSV = os.path.join(RESULT_DIR, "clinical_labels_simple.csv")
OUTPUT_TABLE = os.path.join(RESULT_DIR, "RISK_Univariate_Analysis.csv")
OUTPUT_FIG = os.path.join(RESULT_DIR, "Supp_FigureS1_RISK_Boxplots.png")

# 设置全局字体大小
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 22

def main():
    print("="*60)
    print("📊 步骤02：RISK特征单变量分析")
    print("="*60)
    
    df_feat = pd.read_csv(FEATURES_CSV)
    df_label = pd.read_csv(LABELS_CSV)
    data = pd.merge(df_feat, df_label, on='Patient_ID')
    
    risk_cols = [c for c in data.columns if c.startswith('RISK_')]
    risk_data = data[risk_cols + ['Label']]
    
    print(f"\nRISK特征列表: {risk_cols}")
    print(f"高风险组: {risk_data['Label'].sum()} 例")
    print(f"低风险组: {len(risk_data) - risk_data['Label'].sum()} 例")
    
    print("\n" + "-"*50)
    print("Mann-Whitney U检验（高风险 vs 低风险）:")
    print("-"*50)
    
    results = []
    for col in risk_cols:
        high_risk = risk_data[risk_data['Label'] == 1][col].dropna()
        low_risk = risk_data[risk_data['Label'] == 0][col].dropna()
        
        if len(high_risk) > 0 and len(low_risk) > 0:
            stat, p = mannwhitneyu(high_risk, low_risk, alternative='two-sided')
            median_high = high_risk.median()
            median_low = low_risk.median()
            ratio = median_high / (median_low + 1e-8)
            
            results.append({
                'Feature': col,
                'High_Risk_Median': median_high,
                'Low_Risk_Median': median_low,
                'Ratio': ratio,
                'p_value': p,
                'Significant': p < 0.05
            })
            
            sig_mark = "✓✓✓" if p < 0.01 else "✓✓" if p < 0.05 else "✓" if p < 0.1 else "✗"
            print(f"  {sig_mark} {col:<25}: 高风险={median_high:.6f}, 低风险={median_low:.6f}, p={p:.4f}")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_TABLE, index=False)
    
    # 绘制箱线图
    risk_cols_plot = risk_cols[:min(6, len(risk_cols))]
    if len(risk_cols_plot) > 0:
        # 增大图形尺寸
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()
        
        for i, col in enumerate(risk_cols_plot):
            ax = axes[i]
            
            # 准备数据
            data_to_plot = [
                risk_data[risk_data['Label'] == 0][col].dropna(),
                risk_data[risk_data['Label'] == 1][col].dropna()
            ]
            
            # 绘制箱线图
            bp = ax.boxplot(data_to_plot, labels=['Low Risk', 'High Risk'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#2E86AB')
            bp['boxes'][1].set_facecolor('#E74C3C')
            
            # 设置标题（含 p 值）
            p_val = results[i]['p_value']
            title = col.replace('RISK_', '')
            if p_val < 0.05:
                title += f"\n(p = {p_val:.3f}*)"
            ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
            
            # 设置轴标签
            ax.set_ylabel('Value', fontsize=18, fontweight='bold', labelpad=10)
            ax.set_xlabel('Risk Group', fontsize=18, fontweight='bold', labelpad=10)
            ax.tick_params(axis='both', labelsize=16, width=2, length=6)
            
            # 设置 y 轴网格线
            ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        
        # 隐藏多余的子图
        for j in range(i+1, 6):
            axes[j].set_visible(False)
        
        # 总标题
        plt.suptitle('RISK Features: High vs Low Risk Groups', fontsize=26, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 箱线图保存至: {OUTPUT_FIG}")
    
    print(f"\n💾 结果保存至: {OUTPUT_TABLE}")
    print("="*60)

if __name__ == "__main__":
    main()