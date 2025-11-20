import torch
import os
import pandas as pd
from config import ModelConfig
from scaler_manager import ScalerManager
from physics_model import EnhancedPhysicsInformedThermocline
from save_utils import PredictionSaver
from data_loader import load_csv_data_from_df
from compute_approximate_velocity import compute_velocity

# === 路径设置 ===
input_path = "/Users/yidu/Desktop/MathSim/data-mean/processed_data_mean.csv"
output_dir = "/Users/yidu/Desktop/EnhancedPINN/Test0.18"
model_path = os.path.join(output_dir, "model_final.pt")
os.makedirs(output_dir, exist_ok=True)

# === 加载数据 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(input_path)
t, x, y, z, u_true, v_true, scaler_mgr, original_df = load_csv_data_from_df(df, device)

# === 初始化模型配置 ===
config = ModelConfig(
    Ro=0.024,  # 初始化值会被权重覆盖
    omega_0=30.0,
    use_scaler=True,
    scaler_mgr=scaler_mgr,
    depth_scaler=scaler_mgr.depth_scaler,
    velocity_scale=1.0,
    grad_clip=10.0
)

# === 构建模型并加载权重 ===
model = EnhancedPhysicsInformedThermocline(config).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# ✅ 检查 Ro 是否加载成功
print(f"✅ Ro used for prediction: {model.Ro.item():.6f}")

# === 执行预测 ===
x.requires_grad_(True)
y.requires_grad_(True)
u_pred, v_pred, _, _, _, _, Z0, _, g1, g2, *_ = compute_velocity(model, x, y, z, t)

# === 保存结果 ===
final_losses = {}  # 如果你有损失项可以填入
saver = PredictionSaver(model, scaler_mgr, output_dir)
saver.save_all(
    original_df=original_df,
    u_pred=u_pred, v_pred=v_pred,
    u_true=u_true, v_true=v_true,
    Z0=Z0, x=x, y=y, z=z, t=t,
    g1=g1, g2=g2,
    final_losses=final_losses,
    extra_metrics={"Ro": model.Ro.item()}
)

print("✅ 预测完成，结果已保存。")
