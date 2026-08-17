# 09_Temporal_Evolution_Analysis.py
"""
📈 步骤 08：纵向演变趋势分析 (Temporal Evolution Analysis)
- 目标：展示核心证据特征 'BIOMARKER_glcm_DifferenceVariance' 随治疗分次的演变趋势。
- 意义：证明高风险患者在治疗早期即表现出特定不确定性特征的异常。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ================== 配置 ==================
RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
FEATURES_CSV = os.path.join(RESULT_DIR, "Fractional_Evidential_Features.csv")
LABELS_CSV = os.path.join(RESULT_DIR, "clinical_labels_simple.csv")
OUTPUT_FIG = os.path.join(RESULT_DIR, "Main_Figure5_Temporal.png")

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

def main():
    print("\n" + "="*75)
    print("📈 步骤 08：纵向演变趋势分析 (冠军特征追踪)")
    print("="*75)

    # 1. 加载并合并数据
    if not os.path.exists(FEATURES_CSV):
        print(f"❌ 找不到特征文件: {FEATURES_CSV}")
        return

    df_feat = pd.read_csv(FEATURES_CSV)
    df_label = pd.read_csv(LABELS_CSV)
    data = pd.merge(df_feat, df_label, on='Patient_ID')

    # 2. 提取分次数字 (例如 "Fr_1" -> 1)
    data['Fraction_Num'] = data['Fraction_ID'].apply(
        lambda x: int(x.split('_')[-1]) if isinstance(x, str) and '_' in x else 1
    )

    # 3. 指定绘图目标 (步骤 07 发现的最强特征)
    target_feature = 'BIOMARKER_glcm_DifferenceVariance'
    
    if target_feature not in data.columns:
        # 容错：如果列名不匹配，列出所有可用特征
        print(f"⚠️ 找不到特征 {target_feature}，正在尝试自动匹配...")
        match_cols = [c for c in data.columns if 'DifferenceVariance' in c]
        if match_cols:
            target_feature = match_cols[0]
            print(f"✅ 已自动匹配到: {target_feature}")
        else:
            print(f"❌ 错误：在 CSV 中找不到相关特征。")
            return

    # 4. 开始绘图
    plt.figure(figsize=(9, 6))

    # 绘制带置信区间的折线图
    # ci=68 表示标准误 (Standard Error), palette 颜色：红色代表高风险，蓝色代表低风险
    ax = sns.lineplot(
        x='Fraction_Num', 
        y=target_feature, 
        hue='Label', 
        data=data,
        marker='o', 
        linewidth=3, 
        markersize=10,
        palette={0: '#3498DB', 1: '#E74C3C'},
        errorbar=('ci', 68) 
    )

    # 5. 美化图表
    plt.title(f'Temporal Evolution of Evidential Biomarker\n({target_feature})', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Treatment Fraction Number', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Intensity (Normalized)', fontsize=12, fontweight='bold')
    
    # 确保 X 轴刻度为整数
    max_fr = data['Fraction_Num'].max()
    plt.xticks(np.arange(1, max_fr + 1))

    # 修改图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Low Risk (Label 0)', 'High Risk (Label 1)'], 
              title='Clinical Toxicity Status', title_fontsize='11', loc='best')

    # 6. 添加医学物理解读标注 (修复箭头越界问题)
    y_high_mean = data[data['Label']==1][target_feature].mean()
    plt.annotate('Dose Spillage\nPattern Separation', 
                 xy=(max_fr, y_high_mean), 
                 xytext=(max_fr - 0.8, y_high_mean - 0.6), # 将文字固定在数据点左下方附近
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=11, fontstyle='italic', ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
				 
    plt.tight_layout()
    
    # 7. 保存结果
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n📊 分析结论：")
    print(f"   - 绘制特征: {target_feature}")
    print(f"   - 高风险组均值 (全分次): {data[data['Label']==1][target_feature].mean():.4f}")
    print(f"   - 低风险组均值 (全分次): {data[data['Label']==0][target_feature].mean():.4f}")
    print(f"\n✅ 纵向趋势图已保存至: {OUTPUT_FIG}")
    print("="*75)

if __name__ == "__main__":
    main()
