# 03_Extract_Longitudinal_Features.py (完整版 - 双模式 + RISK 特征)
"""
纵向证据组学特征提取 - 一次运行同时生成半影区和全器官两种模式
同时提取 RISK 特征用于单变量分析
- 半影区 ROI (20%-80% Dmax) → Fractional_Evidential_Features.csv
- 全器官 ROI (>5% Dmax) → Fractional_Global_Features.csv (消融实验对照)
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.ndimage import sobel, gaussian_filter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PREV_RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\data"
OUTPUT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SETTINGS = {
    'binWidth': 5,
    'interpolator': 'sitkBSpline',
    'resampledPixelSpacing': [2.0, 2.0, 2.0],
    'force2D': False,
    'normalize': False,
}

def create_sitk_image(arr, spacing):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing([float(s) for s in spacing])
    return img

def compute_dose_gradient(dose_3d, sigma=1.0):
    dose_3d = dose_3d.astype(np.float32)
    dose_smooth = gaussian_filter(dose_3d, sigma=sigma)
    grad_x = sobel(dose_smooth, axis=0)
    grad_y = sobel(dose_smooth, axis=1)
    grad_z = sobel(dose_smooth, axis=2)
    return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

def extract_risk_features(risk_map, organ_mask):
    """从 risk_map 中提取统计特征"""
    risk_vals = risk_map[organ_mask > 0]
    if len(risk_vals) == 0:
        return {}
    
    return {
        'RISK_Volume_Fraction': float(np.sum(organ_mask) / organ_mask.size),
        'RISK_Mean': float(np.mean(risk_vals)),
        'RISK_Max': float(np.max(risk_vals)),
        'RISK_Median': float(np.median(risk_vals)),
        'RISK_90th': float(np.percentile(risk_vals, 90)),
        'RISK_Entropy_Norm': float(-np.sum(risk_vals * np.log(risk_vals + 1e-8)) / len(risk_vals))
    }

def extract_features_for_fraction(npz_path, pid, fr_id, roi_mode='penumbra'):
    """
    提取特征
    roi_mode: 'penumbra' (半影区 20%-80%) 或 'global' (全器官 >5%)
    """
    if not os.path.exists(npz_path): 
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        dose_warped = data.get('warped_dose', data.get('warped_ct', None))
        uncertainty = data.get('uncertainty', None)
        
        if dose_warped is None or uncertainty is None: 
            return None
            
        dose_warped = np.nan_to_num(dose_warped.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        uncertainty = np.nan_to_num(uncertainty.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        spacing = data.get('spacing', (2.0, 2.0, 2.0))
        if isinstance(spacing, np.ndarray): 
            spacing = spacing.tolist()
        
        dose_max = dose_warped.max()
        if dose_max < 1e-6: 
            return None

        # ========== 根据模式选择 ROI ==========
        if roi_mode == 'penumbra':
            # 半影区：20%-80% Dmax
            lower_bound = dose_max * 0.20
            upper_bound = dose_max * 0.80
            organ_mask = ((dose_warped >= lower_bound) & (dose_warped <= upper_bound)).astype(np.uint8)
            if np.sum(organ_mask) < 200:
                organ_mask = (dose_warped > (dose_max * 0.1)).astype(np.uint8)
        else:  # global mode
            # 全器官：>5% Dmax (消融实验)
            organ_mask = (dose_warped > (dose_max * 0.05)).astype(np.uint8)
        
        if np.sum(organ_mask) < 50: 
            return None

        # 不确定性归一化
        unc_min, unc_max = uncertainty.min(), uncertainty.max()
        if (unc_max - unc_min) > 1e-8:
            unc_norm = (uncertainty - unc_min) / (unc_max - unc_min + 1e-8)
        else:
            unc_norm = np.zeros_like(uncertainty)
            
        dose_grad = compute_dose_gradient(dose_warped)
        
        # 🔴 计算 RISK 特征（用于单变量分析）
        risk_map = dose_grad * unc_norm
        risk_features = extract_risk_features(risk_map, organ_mask)
        
        # 使用 85% 不确定性阈值分离 BIO 和 CLEAN
        threshold = np.percentile(unc_norm[organ_mask > 0], 85)
        mask_high = ((unc_norm >= threshold) & (organ_mask > 0)).astype(np.uint8)
        mask_low = ((unc_norm < threshold) & (organ_mask > 0)).astype(np.uint8)
        
        extractor = featureextractor.RadiomicsFeatureExtractor(**SETTINGS)
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        
        features = {"Patient_ID": pid, "Fraction_ID": fr_id, "Sample_ID": f"{pid}_{fr_id}"}
        sitk_dose = create_sitk_image(dose_warped, spacing)
        
        # CLEAN 特征 (低不确定性区域)
        if np.sum(mask_low) > 20:
            f_low = extractor.execute(sitk_dose, create_sitk_image(mask_low, spacing))
            for k, v in f_low.items():
                if not k.startswith('diagnostics'): 
                    features[f"CLEAN_{k.replace('original_', '')}"] = float(v)
        
        # BIOMARKER 特征 (高不确定性区域)
        if np.sum(mask_high) > 20:
            f_high = extractor.execute(sitk_dose, create_sitk_image(mask_high, spacing))
            for k, v in f_high.items():
                if not k.startswith('diagnostics'): 
                    features[f"BIOMARKER_{k.replace('original_', '')}"] = float(v)
        
        # 🔴 添加 RISK 特征
        features.update(risk_features)
        
        return features
    except Exception as e:
        return None

def process_mode(roi_mode, output_csv, mode_name):
    """处理单个 ROI 模式"""
    print(f"\n{'='*60}")
    print(f"🌓 处理模式: {mode_name}")
    print(f"   输出文件: {output_csv}")
    print(f"{'='*60}")
    
    all_records = []
    patients = sorted([d for d in os.listdir(PREV_RESULT_DIR) 
                      if os.path.isdir(os.path.join(PREV_RESULT_DIR, d))])
    
    for pid in tqdm(patients, desc=f"处理患者 ({mode_name})"):
        p_dir = os.path.join(PREV_RESULT_DIR, pid)
        fractions = [f for f in os.listdir(p_dir) 
                    if os.path.isdir(os.path.join(p_dir, f)) and f.lower().startswith("fr_")]
        
        if not fractions:
            npz = os.path.join(p_dir, "results.npz")
            feat = extract_features_for_fraction(npz, pid, "Single", roi_mode=roi_mode)
            if feat: 
                all_records.append(feat)
        else:
            for fr in fractions:
                npz = os.path.join(p_dir, fr, "results.npz")
                feat = extract_features_for_fraction(npz, pid, fr, roi_mode=roi_mode)
                if feat: 
                    all_records.append(feat)
    
    if not all_records:
        print(f"❌ {mode_name}: 无有效记录")
        return False
    
    df = pd.DataFrame(all_records)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df.to_csv(output_csv, index=False)
    
    print(f"✅ {mode_name}: 特征提取成功！有效样本数: {len(df)}")
    return True

def main():
    print("="*70)
    print("📊 纵向证据组学特征提取 (双模式并行版 + RISK 特征)")
    print("   将同时生成半影区和全器官两种 ROI 特征")
    print("="*70)
    
    # 定义两种模式
    modes = [
        {
            'roi_mode': 'penumbra',
            'output_csv': os.path.join(OUTPUT_DIR, "Fractional_Evidential_Features.csv"),
            'name': '半影区 ROI (20%-80% Dmax) - 主结果'
        },
        {
            'roi_mode': 'global',
            'output_csv': os.path.join(OUTPUT_DIR, "Fractional_Global_Features.csv"),
            'name': '全器官 ROI (>5% Dmax) - 消融实验对照'
        }
    ]
    
    # 依次处理两种模式
    for mode in modes:
        process_mode(mode['roi_mode'], mode['output_csv'], mode['name'])
    
    print("\n" + "="*70)
    print("✅ 所有特征提取完成！")
    print(f"   主结果文件: {os.path.join(OUTPUT_DIR, 'Fractional_Evidential_Features.csv')}")
    print(f"   消融对照文件: {os.path.join(OUTPUT_DIR, 'Fractional_Global_Features.csv')}")
    print("="*70)

if __name__ == "__main__":
    main()