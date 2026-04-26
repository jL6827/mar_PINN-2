#!/usr/bin/env python3
# plot_thermocline_surface.py
# 说明：
#   使用已训练好的模型在测试集 (x,y,t) 的散点位置上计算温跃层/界面深度场（这里取模型输出的 Z0），
#   并用 3D 三角网曲面图 (plot_trisurf) 绘制 Z0(x,y) 的空间分布。
#
#   之所以使用 Z0：在本仓库模型结构中，Z0 是网络输出的“界面/中心深度”场（依赖 t,x,y,z 的 forward 输出），
#   在 compute_velocity(...) 中也会返回，并在速度修正项 decay = exp(-|z-Z0|/Ro) 中作为垂向中心。
#
# 用法示例：
#   python plot_thermocline_surface.py \
#     --model outputs/exp_1/model_final.pt \
#     --train-csv data/processed_data_mean_train.csv \
#     --test-csv data/processed_data_mean_test.csv \
#     --outdir outputs/exp_1 \
#     --device cpu
#
# 输出：
#   <outdir>/fig/thermocline_Z0_trisurf.png

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # 避免无 GUI 环境报错
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from data_loader import load_csv_data_from_df
from config import ModelConfig
from physics_model import EnhancedPhysicsInformedThermocline


def load_model(model_path: str, scaler_mgr, device: torch.device):
    config = ModelConfig(
        Ro=0.024,
        omega_0=30.0,
        use_scaler=True,
        scaler_mgr=scaler_mgr,
        depth_scaler=scaler_mgr.depth_scaler,
        velocity_scale=1.0,
        grad_clip=10.0,
    )
    model = EnhancedPhysicsInformedThermocline(config)

    sd = torch.load(model_path, map_location=device)
    # 兼容 DataParallel 保存的 module. 前缀
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    try:
        model.load_state_dict(sd)
    except Exception:
        if isinstance(sd, dict) and "state_dict" in sd:
            state = sd["state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state)
        else:
            raise

    model.to(device)
    model.eval()
    return model


def plot_trisurf_z0(x: np.ndarray, y: np.ndarray, z0: np.ndarray, save_path: str, title: str):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z0 = np.asarray(z0, dtype=float).ravel()
    if not (x.shape == y.shape == z0.shape):
        raise ValueError(f"shape 不一致：x={x.shape}, y={y.shape}, z0={z0.shape}")

    # 三角剖分（适用于散点）
    tri = Triangulation(x, y)

    fig = plt.figure(figsize=(9, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_trisurf(tri, z0, cmap="viridis", linewidth=0.2, antialiased=True, alpha=0.95)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="Z0 (thermocline depth proxy)")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Z0")

    # 视角可以按需要调整
    ax.view_init(elev=30, azim=-135)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="绘制温跃层界面深度曲面图（散点三角网），使用模型输出 Z0 作为界面深度代理。")
    p.add_argument("--model", required=True, help="已训练模型路径，例如 outputs/exp_1/model_final.pt")
    p.add_argument("--train-csv", required=True, help="训练集 CSV（用于拟合 scaler，与训练一致）")
    p.add_argument("--test-csv", required=True, help="测试集 CSV（提供散点 x,y,t 等）")
    p.add_argument("--outdir", required=True, help="输出目录（将在其下创建 fig 文件夹）")
    p.add_argument("--device", default="cpu", help="设备，例如 cpu 或 cuda:0")
    p.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="绘图散点最大数量（过大会很慢）。默认 5000；设为 0 表示用全量。",
    )
    p.add_argument(
        "--title",
        default="Thermocline interface (Z0) over (x,y) samples",
        help="图标题",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.outdir)
    fig_dir = outdir / "fig"
    save_path = fig_dir / "thermocline_Z0_trisurf.png"

    # 1) fit scaler
    train_df = pd.read_csv(args.train_csv)
    _, _, _, _, _, _, scaler_mgr, _ = load_csv_data_from_df(train_df, device=device, fit_scaler=True)

    # 2) load test + transform with scaler
    test_df = pd.read_csv(args.test_csv)
    t, x, y, z, u_true, v_true, _, original_test_df = load_csv_data_from_df(
        test_df, device=device, scaler_mgr=scaler_mgr, fit_scaler=False
    )

    # 3) optional sampling (for speed)
    n = x.shape[0]
    if args.sample is not None and args.sample > 0 and n > args.sample:
        idx = torch.randperm(n, device=device)[: args.sample]
        t_s = t[idx]
        x_s = x[idx]
        y_s = y[idx]
        # Z0 理论上不依赖 z，但模型 forward 里 Z0 的输出来自 (t,x,y,z)；这里给一个固定 z=0（归一化空间）
        # 这样 Z0 只体现 t,x,y 的变化。若你希望用真实深度点，也可以换成 z[idx]。
        z_s = torch.zeros_like(x_s)
    else:
        t_s, x_s, y_s = t, x, y
        z_s = torch.zeros_like(x)

    # 4) load model
    model = load_model(args.model, scaler_mgr, device)

    # 5) compute Z0
    with torch.no_grad():
        out = model.forward(t_s, x_s, y_s, z_s)
        z0 = out[:, 9:10]

    # 6) inverse transform x,y,depth to physical units (optional but recommended)
    # scaler_manager 提供了 inverse_x / inverse_y / inverse_transform_depth
    try:
        x_phys = scaler_mgr.inverse_x(x_s).reshape(-1)
        y_phys = scaler_mgr.inverse_y(y_s).reshape(-1)
    except Exception:
        # 如果 scaler 不支持，就退回归一化坐标
        x_phys = x_s.detach().cpu().numpy().reshape(-1)
        y_phys = y_s.detach().cpu().numpy().reshape(-1)

    try:
        z0_phys = scaler_mgr.inverse_transform_depth(z0).reshape(-1)
    except Exception:
        z0_phys = z0.detach().cpu().numpy().reshape(-1)

    plot_trisurf_z0(x_phys, y_phys, z0_phys, str(save_path), args.title)
    print(f"已保存温跃层曲面图: {save_path}")


if __name__ == "__main__":
    main()
