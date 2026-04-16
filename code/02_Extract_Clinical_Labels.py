# 02_Extract_Clinical_Labels.py
"""
剂量学标签提取：基于十二指肠V33和胃V40的联合阈值
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = r"D:\0临床科研\剂量学衰退\data\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
OUTPUT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "clinical_labels.csv")
OUTPUT_SIMPLE = os.path.join(OUTPUT_DIR, "clinical_labels_simple.csv")

def find_rtdose_file(patient_path):
    for root, dirs, files in os.walk(patient_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                try:
                    ds = pydicom.dcmread(os.path.join(root, file), stop_before_pixels=True)
                    if getattr(ds, 'Modality', '') == 'RTDOSE':
                        return os.path.join(root, file)
                except:
                    continue
    return None

def read_dose_with_normalization(rtdose_path):
    try:
        dose_img = sitk.ReadImage(rtdose_path)
        dose_arr = sitk.GetArrayFromImage(dose_img).astype(np.float32)
        spacing = dose_img.GetSpacing()
        voxel_volume_cc = (spacing[0] * spacing[1] * spacing[2]) / 1000
        
        ds = pydicom.dcmread(rtdose_path)
        dose_scaling = getattr(ds, 'DoseGridScaling', 1.0)
        dose_gy = np.clip(dose_arr * dose_scaling, 0, 100)
        return dose_gy, voxel_volume_cc
    except Exception as e:
        return None, None

def main():
    print("="*60)
    print("📊 步骤02：剂量学标签提取")
    print("="*60)
    
    patients = [p for p in os.listdir(DATA_ROOT) 
                if os.path.isdir(os.path.join(DATA_ROOT, p))]
    print(f"发现 {len(patients)} 个患者")
    
    temp_results = []
    
    for pid in patients:
        rtdose_path = find_rtdose_file(os.path.join(DATA_ROOT, pid))
        if not rtdose_path:
            print(f"  ⚠️ {pid}: 未找到RTDOSE文件")
            continue
        
        dose_gy, voxel_volume_cc = read_dose_with_normalization(rtdose_path)
        if dose_gy is None:
            print(f"  ⚠️ {pid}: 读取剂量失败")
            continue
        
        v33_total = np.sum(dose_gy >= 33) * voxel_volume_cc
        v40_total = np.sum(dose_gy >= 40) * voxel_volume_cc
        
        temp_results.append({
            'Patient_ID': pid,
            'V33_Duodenum_cc': v33_total,
            'V40_Stomach_cc': v40_total
        })
        print(f"  ✓ {pid}: V33={v33_total:.1f}cc, V40={v40_total:.1f}cc")
    
    if not temp_results:
        print("❌ 未提取到任何数据")
        return
    
    df = pd.DataFrame(temp_results)
    
    median_v33 = df['V33_Duodenum_cc'].median()
    median_v40 = df['V40_Stomach_cc'].median()
    
    df['Label'] = ((df['V33_Duodenum_cc'] > median_v33) | 
                   (df['V40_Stomach_cc'] > median_v40)).astype(int)
    
    print(f"\n📊 统计：")
    print(f"   十二指肠 V33 中位数: {median_v33:.1f} cc")
    print(f"   胃 V40 中位数: {median_v40:.1f} cc")
    print(f"   高风险(Label=1): {df['Label'].sum()} 例")
    print(f"   低风险(Label=0): {len(df) - df['Label'].sum()} 例")
    
    df.to_csv(OUTPUT_PATH, index=False)
    df[['Patient_ID', 'Label']].to_csv(OUTPUT_SIMPLE, index=False)
    
    print(f"\n✅ 保存至: {OUTPUT_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()