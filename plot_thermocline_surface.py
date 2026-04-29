#!/usr/bin/env python3
# plot_thermocline_surface.py
# 说明：
#   使用已训练好的模型在测试集 (lon,lat,t) 的散点位置上计算温跃层/界面深度场（这里取模型输出的 Z0），
#   并绘制更“平滑”的二维颜色图（基于散点插值到规则网格）以及可选的 3D 曲面图。
#
#   为什么用 Z0：在本仓库模型结构中，Z0 是网络输出的“界面/中心深度”场，
#   在 compute_velocity(...) 返回，并在 decay = exp(-|z-Z0|/Ro) 中作为垂向中心。
#
# 用法示例：
#   python .\plot_thermocline_surface.py --model outputs/exp_7/model_final.pt --train-csv data/processed_data_mean_train.csv --test-csv data/processed_data_mean_test.csv --outdir outputs/exp_7 --device cuda:0 --sample 0 --grid-nx 1000 --grid-ny 1000 --cmap jet --levels 256
#
# 输出：
#   <outdir>/fig/thermocline_Z0_contourf.png   (推荐：更平滑)
#   <outdir>/fig/thermocline_Z0_surface.png    (可选：3D)

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")  # 避免无 GUI 环境报错
import matplotlib.pyplot as plt

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

    # 显式设置 weights_only，避免 FutureWarning；如果你的 torch 版本不支持，会自动回退。
    try:
        sd = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
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


def interpolate_to_grid(x, y, z, nx=200, ny=200, method="linear"):
    """把散点 (x,y,z) 插值到规则网格 (X,Y,Zg)。优先用 SciPy；没有 SciPy 时用 matplotlib.tri。"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    xi = np.linspace(np.nanmin(x), np.nanmax(x), nx)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), ny)
    X, Y = np.meshgrid(xi, yi)

    # 1) SciPy griddata（最平滑、可控）
    try:
        from scipy.interpolate import griddata  # type: ignore
        Zg = griddata((x, y), z, (X, Y), method=method)
        return X, Y, Zg
    except Exception:
        pass

    # 2) fallback: matplotlib Triangulation + LinearTriInterpolator
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(x, y)
    if method == "nearest":
        interp = mtri.CubicTriInterpolator(tri, z)  # 没有真正 nearest，这里给一个近似更平滑的
    else:
        # linear / cubic（cubic 可能更平滑，但在稀疏区域可能产生振荡）
        interp = mtri.LinearTriInterpolator(tri, z) if method == "linear" else mtri.CubicTriInterpolator(tri, z)

    Zg = interp(X, Y)
    # 可能是 masked array
    if hasattr(Zg, "filled"):
        Zg = Zg.filled(np.nan)
    return X, Y, Zg


def plot_contourf(
    X,
    Y,
    Z,
    save_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    cmap: str = "viridis",
    levels=30,
):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

    # levels 越多，颜色过渡越连续（文件可能更大）
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_surface(X, Y, Z, save_path: str, title: str, xlabel: str, ylabel: str, zlabel: str, cmap: str = "viridis"):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Z 中的 nan 会导致 surface 报错，做个 mask
    Zm = np.ma.masked_invalid(Z)
    surf = ax.plot_surface(X, Y, Zm, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label=zlabel)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=30, azim=-135)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="绘制温跃层界面深度图（使用模型输出 Z0 作为界面深度代理）。")
    p.add_argument("--model", required=True, help="已训练模型路径，例如 outputs/exp_1/model_final.pt")
    p.add_argument("--train-csv", required=True, help="训练集 CSV（用于拟合 scaler，与训练一致）")
    p.add_argument("--test-csv", required=True, help="测试集 CSV（提供散点经纬度等）")
    p.add_argument("--outdir", required=True, help="输出目录（将在其下创建 fig 文件夹）")
    p.add_argument("--device", default="cpu", help="设备，例如 cpu 或 cuda:0")
    p.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="用于绘图/插值的散点最大数量。默认 5000；设为 0 表示用全量（可能很慢）。",
    )
    p.add_argument("--grid-nx", type=int, default=250, help="插值网格 x 方向分辨率")
    p.add_argument("--grid-ny", type=int, default=250, help="插值网格 y 方向分辨率")
    p.add_argument(
        "--interp",
        choices=["linear", "cubic", "nearest"],
        default="linear",
        help="散点到网格插值方法（有 SciPy 时更准确）。",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="matplotlib colormap 名称，例如 jet / turbo / viridis / plasma ...（想要 MATLAB jet 就用 jet）",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=30,
        help="contourf 的颜色等级数；越大颜色过渡越平滑，例如 128 或 256。",
    )
    p.add_argument(
        "--plot-3d",
        action="store_true",
        help="额外输出 3D surface 图（默认只输出更平滑的 2D contourf）。",
    )
    p.add_argument(
        "--title",
        default="Thermocline interface depth (Z0)",
        help="图标题",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.outdir)
    fig_dir = outdir / "fig"
    save_contour = fig_dir / "thermocline_Z0_contourf.png"
    save_surface = fig_dir / "thermocline_Z0_surface.png"

    # 1) fit scaler
    train_df = pd.read_csv(args.train_csv)
    _, _, _, _, _, _, scaler_mgr, _ = load_csv_data_from_df(train_df, device=device, fit_scaler=True)

    # 2) load test + transform with scaler
    test_df = pd.read_csv(args.test_csv)
    t, x, y, z, _, _, _, _ = load_csv_data_from_df(test_df, device=device, scaler_mgr=scaler_mgr, fit_scaler=False)

    # 3) optional sampling
    n = x.shape[0]
    if args.sample is not None and args.sample > 0 and n > args.sample:
        idx = torch.randperm(n, device=device)[: args.sample]
        t_s = t[idx]
        x_s = x[idx]
        y_s = y[idx]
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

    # 6) inverse to physical units
    # 在 scaler_manager.fit 中，x/y 对应列是 longitude/latitude
    try:
        lon = scaler_mgr.inverse_x(x_s).reshape(-1)
        lat = scaler_mgr.inverse_y(y_s).reshape(-1)
        x_label = "Longitude"
        y_label = "Latitude"
    except Exception:
        lon = x_s.detach().cpu().numpy().reshape(-1)
        lat = y_s.detach().cpu().numpy().reshape(-1)
        x_label = "x (normalized)"
        y_label = "y (normalized)"

    try:
        z0_depth = scaler_mgr.inverse_transform_depth(z0).reshape(-1)
        z_label = "Depth (m)"
    except Exception:
        z0_depth = z0.detach().cpu().numpy().reshape(-1)
        z_label = "Z0"

    # 7) interpolate to grid (smooth)
    Xg, Yg, Zg = interpolate_to_grid(lon, lat, z0_depth, nx=args.grid_nx, ny=args.grid_ny, method=args.interp)

    # ========== 新增：导出网格数据到.mat文件 ==========
    from scipy.io import savemat

    # 定义.mat文件保存路径（和图片同目录）
    mat_save_path = fig_dir / "thermocline_grid_data.mat"
    # 保存数据（MATLAB 读取时变量名对应：Xg, Yg, Zg, 轴标签）
    savemat(
        str(mat_save_path),
        {
            "Xg": Xg,          # 经度网格 (ny, nx)
            "Yg": Yg,          # 纬度网格 (ny, nx)
            "Zg": Zg,          # 深度数据 (ny, nx)
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
            "title_str": args.title
        }
    )
    print(f"已导出 MATLAB 绘图数据: {mat_save_path}")
    # ================================================

    # 8) plot 2D smooth contourf
    plot_contourf(
        Xg,
        Yg,
        Zg,
        str(save_contour),
        title=args.title,
        xlabel=x_label,
        ylabel=y_label,
        cbar_label=z_label,
        cmap=args.cmap,
        levels=args.levels,
    )
    print(f"已保存平滑 2D 图: {save_contour}")

    # optional 3D surface
    if args.plot_3d:
        plot_surface(
            Xg,
            Yg,
            Zg,
            str(save_surface),
            title=args.title,
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
            cmap=args.cmap,
        )
        print(f"已保存 3D surface 图: {save_surface}")


if __name__ == "__main__":
    main()