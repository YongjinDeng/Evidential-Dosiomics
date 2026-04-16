# 01_PI_INR_Longitudinal.py
"""
全纵向 PI-INR 配准引擎 (MedIA 原版 + 热启动 + 完整物理约束)
- 完整保留 phys_loss (剂量梯度引导的 Jacobian 约束)
- 完整保留 fold_loss (防止图像折叠)
- 整合热启动加速
"""

import os
import gc
import torch
import pydicom
import numpy as np
import SimpleITK as sitk
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ================== 1. 配置 (补全所有参数) ==================
CONFIG = {
    "DATA_ROOT": r"D:\0临床科研\剂量学衰退\data\manifest-1661266724052\Pancreatic-CT-CBCT-SEG",
    "RESULT_DIR": r"D:\0临床科研\胰腺癌毒性\data",
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 配准参数
    "EPOCHS_FR1": 1500,
    "EPOCHS_WARM": 400,
    "BATCH_POINTS": 25000,
    "LR": 5e-4,
    "OMEGA_0": 20.0,
    
    # 损失权重 (完整版)
    "LAMBDA_EDL": 1.0,
    "LAMBDA_SSIM": 0.0,
    "LAMBDA_EDGE": 5.0,
    "LAMBDA_SMOOTH": 0.1,
    "LAMBDA_PHYS": 0.2,           # 物理损失权重
    "LAMBDA_FOLD_INIT": 0.1,
    "LAMBDA_FOLD_MAX": 5.0,
    "GAMMA_METRIC": 10.0,         # 剂量梯度缩放因子
    
    "TARGET_SPACING": (2.0, 2.0, 2.0),
    "RANDOM_SEED": 42,
}

os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["RANDOM_SEED"])

# ================== 2. 网络定义 ==================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=20.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / self.omega_0,
                                            np.sqrt(6 / in_features) / self.omega_0)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class RiemannianSirenNet(nn.Module):
    def __init__(self, omega_0=20.0):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(3, 128, is_first=True, omega_0=omega_0),
            SineLayer(128, 128, omega_0=omega_0),
            SineLayer(128, 128, omega_0=omega_0),
            nn.Linear(128, 6)
        )
        with torch.no_grad():
            self.net[-1].weight.fill_(0)
            self.net[-1].bias.fill_(0)
    
    def forward(self, x):
        out = self.net(x)
        disp = torch.tanh(out[..., :3]) * 0.05
        v = F.softplus(out[..., 3:4]) + 1e-6
        alpha = F.softplus(out[..., 4:5]) + 1.1
        beta = F.softplus(out[..., 5:6]) + 1e-6
        return disp, v.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)

# ================== 3. 核心工具函数 ==================
def compute_image_gradients(img_tensor):
    dz = img_tensor[..., 2:, 1:-1, 1:-1] - img_tensor[..., :-2, 1:-1, 1:-1]
    dy = img_tensor[..., 1:-1, 2:, 1:-1] - img_tensor[..., 1:-1, :-2, 1:-1]
    dx = img_tensor[..., 1:-1, 1:-1, 2:] - img_tensor[..., 1:-1, 1:-1, :-2]
    mag = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-6)
    return F.pad(mag, (1, 1, 1, 1, 1, 1), mode='replicate')

def fast_stratified_sample(batch_size, edge_idx, bg_idx, fixed_flat, fixed_edges_flat, shape, device):
    D, H, W = shape
    e_batch = int(batch_size * 0.7)
    b_batch = batch_size - e_batch
    
    if len(edge_idx) > 0:
        s_edge = edge_idx[torch.randint(0, len(edge_idx), (e_batch,), device=device)]
    else:
        s_edge = bg_idx[torch.randint(0, len(bg_idx), (e_batch,), device=device)]
    
    if len(bg_idx) > 0:
        s_bg = bg_idx[torch.randint(0, len(bg_idx), (b_batch,), device=device)]
    else:
        s_bg = edge_idx[torch.randint(0, len(edge_idx), (b_batch,), device=device)]
    
    idx = torch.cat([s_edge, s_bg])
    
    f_s = fixed_flat[idx]
    fe_s = fixed_edges_flat[idx]
    
    z = idx // (H * W)
    y = (idx % (H * W)) // W
    x = idx % W
    
    coords = torch.stack([x, y, z], dim=1).float()
    sizes = torch.tensor([W - 1, H - 1, D - 1], device=device).float()
    coords = (coords / sizes) * 2.0 - 1.0
    
    return coords.requires_grad_(True), f_s, fe_s

def compute_jacobian_fast(y, x, grad_outputs_list):
    jac = []
    for i in range(3):
        grad = torch.autograd.grad(y[:, i], x, grad_outputs=grad_outputs_list[i],
                                   create_graph=True, retain_graph=True)[0]
        jac.append(grad)
    return torch.stack(jac, dim=1)

def compute_fold_loss(J, I_mat, device):
    det = torch.det(I_mat + J)
    det_clamped = torch.clamp(det, min=-10.0, max=10.0)
    fold_loss = F.relu(-det_clamped + 1e-5).mean()
    expansion_penalty = torch.mean((det_clamped - 1.0) ** 2) * 0.001
    return fold_loss + expansion_penalty

# ================== 4. 数据加载 ==================
def resample_to_target_spacing(image, target_spacing, default_value=0):
    orig_spacing, orig_size = image.GetSpacing(), image.GetSize()
    target_size = [max(1, int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)

def load_patient_data(patient_path, target_spacing):
    ct_series, dose_path = [], None
    
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(root, dcm_files[0]), stop_before_pixels=True)
            modality = getattr(ds, "Modality", "").upper()
            if modality == 'CT' and len(dcm_files) > 20:
                ct_series.append(root)
            elif modality == 'RTDOSE':
                dose_path = os.path.join(root, dcm_files[0])
        except:
            continue
    
    ct_series.sort()
    if len(ct_series) < 2:
        return None
    
    reader = sitk.ImageSeriesReader()
    
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[0]))
    fixed = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_series[1]))
    moving = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    
    def sanitize_image(img, fill_val=-1000):
        arr = np.nan_to_num(sitk.GetArrayFromImage(img), nan=fill_val)
        sanitized = sitk.GetImageFromArray(arr)
        sanitized.CopyInformation(img)
        return sanitized
    
    fixed = sanitize_image(fixed)
    moving = sanitize_image(moving)
    
    fixed_center = np.array(fixed.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in fixed.GetSize()]))
    moving_center = np.array(moving.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in moving.GetSize()]))
    translation = (moving_center - fixed_center).tolist()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(sitk.TranslationTransform(3, translation))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    moving_aligned = resampler.Execute(moving)
    
    if dose_path and os.path.exists(dose_path):
        try:
            resampler.SetTransform(sitk.Transform())
            resampler.SetDefaultPixelValue(0)
            dose_image = resampler.Execute(sanitize_image(sitk.ReadImage(dose_path), 0))
        except:
            dose_image = sanitize_image(sitk.GetImageFromArray(np.zeros(fixed.GetSize()[::-1], dtype=np.float32)), 0)
    else:
        dose_image = sanitize_image(sitk.GetImageFromArray(np.zeros(fixed.GetSize()[::-1], dtype=np.float32)), 0)
    
    fixed = resample_to_target_spacing(fixed, target_spacing, -1000)
    moving_aligned = resample_to_target_spacing(moving_aligned, target_spacing, -1000)
    dose_image = resample_to_target_spacing(dose_image, target_spacing, 0)
    
    def to_tensor(img, is_dose=False):
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if is_dose:
            # 剂量保持原始 Gy 值，不做归一化！
            arr = np.clip(arr, 0, 100)
        else:
            arr = np.clip((arr + 1000) / 2000.0, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(CONFIG["DEVICE"])
    
    return (to_tensor(fixed, False), 
            to_tensor(moving_aligned, False), 
            to_tensor(dose_image, True), 
            fixed, 
            moving_aligned)

# ================== 5. 配准训练函数 (完整损失函数 + phys_loss) ==================
def register_fraction(model, fixed_t, moving_t, dose_t, epochs, save_path=None):
    """完整版训练函数，包含 phys_loss (剂量梯度引导的 Jacobian 约束)"""
    
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-6)
    
    D, H, W = fixed_t.shape[2:]
    
    # 1. 预计算剂量梯度 (灵魂步骤：物理驱动)
    try:
        spacing = [CONFIG["TARGET_SPACING"][2], CONFIG["TARGET_SPACING"][1], CONFIG["TARGET_SPACING"][0]]
        dz, dy, dx = torch.gradient(dose_t.squeeze(), spacing=spacing)
        grad_dose = torch.stack([dx, dy, dz], dim=0).unsqueeze(0).to(CONFIG["DEVICE"])
    except:
        grad_dose = torch.zeros(1, 3, D, H, W, device=CONFIG["DEVICE"])

    fixed_edges = compute_image_gradients(fixed_t)
    fixed_flat = fixed_t.view(-1)
    fixed_edges_flat = fixed_edges.view(-1)
    edge_threshold = fixed_edges_flat.mean()
    
    edge_idx = torch.nonzero(fixed_edges_flat > edge_threshold).squeeze()
    bg_idx = torch.nonzero(fixed_edges_flat <= edge_threshold).squeeze()
    
    if edge_idx.dim() == 0:
        edge_idx = edge_idx.unsqueeze(0)
    if bg_idx.dim() == 0:
        bg_idx = bg_idx.unsqueeze(0)

    # Jacobian 计算需要的工具
    I_mat = torch.eye(3, device=CONFIG["DEVICE"]).unsqueeze(0)
    grad_outputs_list = [torch.ones(CONFIG["BATCH_POINTS"], device=CONFIG["DEVICE"]) for _ in range(3)]
    
    pbar = tqdm(range(epochs), desc=f"Training {epochs} epochs")
    for epoch in pbar:
        opt.zero_grad()
        
        coords, f_s, fe_s = fast_stratified_sample(
            CONFIG["BATCH_POINTS"], edge_idx, bg_idx,
            fixed_flat, fixed_edges_flat, (D, H, W), CONFIG["DEVICE"]
        )
        
        disp, v, alpha, beta = model(coords)
        
        m_s = F.grid_sample(moving_t, (coords + disp).view(1,1,1,-1,3), 
                           align_corners=True, mode='bilinear').view(-1)
        me_s = F.grid_sample(compute_image_gradients(moving_t), (coords + disp).view(1,1,1,-1,3), 
                            align_corners=True, mode='bilinear').view(-1)
        
        mask = (f_s > 0.05).float()
        
        # EDL 损失
        edl_loss = torch.mean(mask * ((f_s - m_s)**2 * v + (2*alpha + v)/(2*alpha*v)))
        
        # 边缘损失
        edge_loss = torch.mean(mask * (fe_s - me_s)**2)
        
        # 物理与折叠损失计算 (恢复 MedIA 设定)
        try:
            J = compute_jacobian_fast(disp, coords, grad_outputs_list)
            s_grad = F.grid_sample(grad_dose, coords.view(1,1,1,-1,3), align_corners=True).view(3,-1).T
            # 剂量梯度引导的物理惩罚
            phys_loss = torch.mean(torch.clamp(1.0 + CONFIG["GAMMA_METRIC"] * torch.sum(s_grad**2, dim=-1), 1.0, 50.0) * torch.sum(J**2, dim=(1,2)))
            fold_loss = compute_fold_loss(J, I_mat, CONFIG["DEVICE"])
        except Exception as e:
            phys_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])
            fold_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])
            
        try:
            grad_disp = torch.autograd.grad(disp.sum(), coords, create_graph=True)[0]
            smooth_loss = torch.mean(grad_disp**2)
        except:
            smooth_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])
            
        progress = epoch / epochs
        w_fold = CONFIG["LAMBDA_FOLD_INIT"] + (CONFIG["LAMBDA_FOLD_MAX"] - CONFIG["LAMBDA_FOLD_INIT"]) * (progress ** 0.5)

        total_loss = (edl_loss + 
                      CONFIG["LAMBDA_EDGE"] * edge_loss + 
                      CONFIG["LAMBDA_SMOOTH"] * smooth_loss + 
                      CONFIG["LAMBDA_PHYS"] * phys_loss + 
                      w_fold * fold_loss)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "edl": f"{edl_loss.item():.4f}",
            "phys": f"{phys_loss.item():.4f}",
            "fold": f"{fold_loss.item():.6f}"
        })
    
    if save_path:
        torch.save(model.state_dict(), save_path)
    
    return model

# ================== 6. 全量推理 ==================
def inference_full_volume(model, fixed_t, moving_t, dose_t, D, H, W):
    """全量体素推理"""
    model.eval()
    displacement = np.zeros((D, H, W, 3), dtype=np.float32)
    uncertainty = np.zeros((D, H, W), dtype=np.float32)
    
    with torch.no_grad():
        for z in tqdm(range(D), desc="Inference"):
            z_n = (z / (D - 1)) * 2 - 1 if D > 1 else 0
            gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=CONFIG["DEVICE"]),
                                    torch.linspace(-1, 1, W, device=CONFIG["DEVICE"]), indexing='ij')
            grid_slice = torch.stack([gx, gy, torch.full_like(gx, z_n)], dim=-1).view(-1, 3)
            disp_s, v_s, alpha_s, beta_s = model(grid_slice)
            displacement[z] = disp_s.view(H, W, 3).cpu().numpy()
            uncertainty[z] = (beta_s / (v_s * (alpha_s - 1))).view(H, W).cpu().numpy()
        
        z_coords = torch.linspace(-1, 1, D, device=CONFIG["DEVICE"])
        y_coords = torch.linspace(-1, 1, H, device=CONFIG["DEVICE"])
        x_coords = torch.linspace(-1, 1, W, device=CONFIG["DEVICE"])
        gz, gy, gx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        base_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)
        
        disp_tensor = torch.from_numpy(displacement).permute(3, 0, 1, 2).unsqueeze(0).to(CONFIG["DEVICE"])
        
        warped_ct = F.grid_sample(moving_t, base_grid + disp_tensor.permute(0, 2, 3, 4, 1),
                                  align_corners=True, mode='bilinear').squeeze().cpu().numpy()
        warped_dose = F.grid_sample(dose_t, base_grid + disp_tensor.permute(0, 2, 3, 4, 1),
                                    align_corners=True, mode='bilinear').squeeze().cpu().numpy()
    
    return displacement, uncertainty, warped_ct, warped_dose

# ================== 7. 主流水线 ==================
def main():
    print("="*70)
    print("🚀 PI-INR 全纵向配准引擎 (MedIA 原版 + 热启动 + 完整物理约束)")
    print(f"   Device: {CONFIG['DEVICE']}")
    print("="*70)
    
    if not os.path.exists(CONFIG["DATA_ROOT"]):
        print(f"❌ 数据路径不存在: {CONFIG['DATA_ROOT']}")
        return
    
    all_patients = sorted([p for p in os.listdir(CONFIG["DATA_ROOT"]) 
                          if p.startswith("Pancreas") and os.path.isdir(os.path.join(CONFIG["DATA_ROOT"], p))])
    
    print(f"发现 {len(all_patients)} 个患者")
    
    for pid in all_patients:
        patient_path = os.path.join(CONFIG["DATA_ROOT"], pid)
        save_root = os.path.join(CONFIG["RESULT_DIR"], pid)
        os.makedirs(save_root, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📂 患者: {pid}")
        print(f"{'='*60}")
        
        data = load_patient_data(patient_path, CONFIG["TARGET_SPACING"])
        if data is None:
            print(f"⚠️ {pid}: 数据加载失败，跳过")
            continue
        
        fixed_t, moving_t, dose_t, fixed_sitk, moving_sitk = data
        D, H, W = fixed_t.shape[2:]
        print(f"   图像尺寸: D={D}, H={H}, W={W}")
        
        # 扫描所有 CT 序列
        ct_series = []
        for root, dirs, files in os.walk(patient_path):
            dcm_files = [f for f in files if f.lower().endswith('.dcm')]
            if not dcm_files:
                continue
            try:
                ds = pydicom.dcmread(os.path.join(root, dcm_files[0]), stop_before_pixels=True)
                if getattr(ds, "Modality", "").upper() == 'CT' and len(dcm_files) > 20:
                    ct_series.append(root)
            except:
                continue
        ct_series.sort()
        
        if len(ct_series) < 2:
            print(f"⚠️ {pid}: 不足 2 个 CT 序列")
            continue
        
        moving_series_list = ct_series[1:]
        print(f"   Moving 分次数: {len(moving_series_list)}")
        
        base_model_path = os.path.join(save_root, "base_warm_start.pth")
        
        for idx, moving_series in enumerate(moving_series_list):
            fr_id = f"Fr_{idx+1}"
            fr_dir = os.path.join(save_root, fr_id)
            os.makedirs(fr_dir, exist_ok=True)
            
            result_path = os.path.join(fr_dir, "results.npz")
            if os.path.exists(result_path):
                file_size = os.path.getsize(result_path)
                if file_size > 50 * 1024 * 1024:
                    print(f"   ⏭️ {fr_id} 已存在，跳过")
                    continue
                else:
                    print(f"   ⚠️ {fr_id} 文件不完整 ({file_size/1024/1024:.1f} MB)，重新处理")
                    os.remove(result_path)
            
            print(f"\n   🔄 处理 {fr_id} ...")
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(moving_series))
            moving_raw = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
            
            def sanitize_image(img, fill_val=-1000):
                arr = np.nan_to_num(sitk.GetArrayFromImage(img), nan=fill_val)
                sanitized = sitk.GetImageFromArray(arr)
                sanitized.CopyInformation(img)
                return sanitized
            
            moving_raw = sanitize_image(moving_raw)
            
            moving_center = np.array(moving_raw.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in moving_raw.GetSize()]))
            fixed_center = np.array(fixed_sitk.TransformContinuousIndexToPhysicalPoint([s/2.0 for s in fixed_sitk.GetSize()]))
            translation = (moving_center - fixed_center).tolist()
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetTransform(sitk.TranslationTransform(3, translation))
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-1000)
            moving_aligned = resampler.Execute(moving_raw)
            
            moving_aligned = resample_to_target_spacing(moving_aligned, CONFIG["TARGET_SPACING"], -1000)
            
            moving_arr = np.clip((sitk.GetArrayFromImage(moving_aligned) + 1000) / 2000.0, 0, 1)
            moving_t_current = torch.from_numpy(moving_arr).unsqueeze(0).unsqueeze(0).to(CONFIG["DEVICE"])
            
            model = RiemannianSirenNet(omega_0=CONFIG["OMEGA_0"]).to(CONFIG["DEVICE"])
            
            if idx > 0 and os.path.exists(base_model_path):
                model.load_state_dict(torch.load(base_model_path, map_location=CONFIG["DEVICE"]))
                epochs = CONFIG["EPOCHS_WARM"]
                print(f"      🔥 热启动模式: {epochs} epochs")
            else:
                epochs = CONFIG["EPOCHS_FR1"]
                print(f"      🆕 完整训练模式: {epochs} epochs")
            
            # 关键：传入 dose_t
            model = register_fraction(model, fixed_t, moving_t_current, dose_t, epochs,
                                      save_path=base_model_path if idx == 0 else None)
            
            displacement, uncertainty, warped_ct, warped_dose = inference_full_volume(
                model, fixed_t, moving_t_current, dose_t, D, H, W
            )
            
            np.savez_compressed(result_path,
                                displacement=displacement.astype(np.float16),
                                uncertainty=uncertainty.astype(np.float16),
                                warped_ct=warped_ct.astype(np.float16),
                                warped_dose=warped_dose.astype(np.float16),
                                shape=(D, H, W),
                                spacing=CONFIG["TARGET_SPACING"])
            
            print(f"      ✅ 已保存: {fr_dir}/results.npz")
            
            del model, moving_t_current, moving_arr, moving_aligned
            torch.cuda.empty_cache()
            gc.collect()
        
        del fixed_t, moving_t, dose_t
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*70)
    print("✅ 全纵向配准完成！")
    print("="*70)

if __name__ == "__main__":
    main()