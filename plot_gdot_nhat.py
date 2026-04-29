#!/usr/bin/env python3
# plot_gdot_nhat.py
# 说明：
#   计算并绘制重力加速度向量 g 与温跃层界面法向单位向量 n? 的点积的空间分布图。
#
# 物理/数学原理（含符号约定）：
#   设温跃层界面为曲面 z = Z0(x, y)（模型输出），将其写成隐式曲面：
#       F(x, y, z) = z - Z0(x, y) = 0
#   则（向上的）法向量为：
#       n = ?F = (-?Z0/?x,  -?Z0/?y,  1)
#   单位法向量（默认取向上，即 z 分量 > 0）：
#       n? = n / ‖n‖，  ‖n‖ = sqrt((?Z0/?x)? + (?Z0/?y)? + 1)
#
#   重力加速度向量（竖直向下为负 z 方向）：
#       g = (0, 0, -g0)，  g0 ≈ 9.81 m/s?
#
#   点积（即重力在法向上的分量）：
#       g · n? = (0·n?_x + 0·n?_y + (-g0)·n?_z)
#             = -g0 · (1 / ‖n‖)
#             = -g0 / sqrt(1 + (?Z0/?x)? + (?Z0/?y)?)
#
#   结果永远为负（约 [-9.81, 0)）：
#     - 界面越平坦（坡度→0），法向越接近竖直，‖n‖→1，点积→-g0（绝对值最大）。
#     - 界面坡度越大，法向越倾斜，‖n‖增大，点积绝对值越小（趋近 0）。
#
#   若指定 --flip-sign（或 --normal-up 逻辑相反方向），则输出 +g0/‖n‖（正值）；
#   这等价于将法向定义为向下（z 分量 < 0）。
#
# 用法示例（exp6 默认值已预填）：
#   python plot_gdot_nhat.py
#   python plot_gdot_nhat.py --flip-sign
#   python plot_gdot_nhat.py \
#     --model outputs/exp_6/model_final.pt \
#     --train-csv data/processed_data_mean_train.csv \
#     --test-csv data/processed_data_mean_test.csv \
#     --outdir outputs/exp_6 \
#     --device cpu \
#     --sample 0 \
#     --cmap RdBu_r \
#     --levels 128
#
# 输出：
#   <outdir>/fig/gdot_nhat_contourf.png
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # 避免无 GUI 环境报错
# 直接复用已有工具函数，保持代码一致性
from plot_thermocline_surface import (
    interpolate_to_grid,
    load_model,
    plot_contourf,
)
from data_loader import load_csv_data_from_df
def compute_gdot_nhat(Zg: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, g0: float = 9.81, flip_sign: bool = False) -> np.ndarray:
    """
    给定规则网格上的界面深度场 Zg = Z0(x, y)，计算重力加速度与界面法向单位向量的点积。
    参数
    ----
    Zg : ndarray, shape (ny, nx)
        插值后规则网格上的界面深度（物理单位，如米）。
    Xg : ndarray, shape (ny, nx)
        规则网格的经度坐标。
    Yg : ndarray, shape (ny, nx)
        规则网格的纬度坐标。
    g0 : float
        重力加速度标量，默认 9.81 m/s?。
    flip_sign : bool
        若为 True，则翻转符号（相当于取向下法向），结果为正值；
        若为 False（默认），法向向上，点积为负值。
    返回
    ----
    gdot : ndarray, shape (ny, nx)
        g · n? 的空间分布，单位与 g0 相同（m/s?）。
        默认（flip_sign=False）范围约 [-9.81, 0)。
        flip_sign=True 时范围约 (0, 9.81]。
    数学说明
    --------
    - 法向量方向：n = (-?Z0/?x, -?Z0/?y, 1)（向上法向，z 分量恒为 +1）
    - ‖n‖ = sqrt((?Z0/?x)? + (?Z0/?y)? + 1)
    - g = (0, 0, -g0)（向下）
    - g · n? = -g0 / ‖n‖
    - 若 flip_sign=True：g · n? = +g0 / ‖n‖
    """
    # 从网格坐标中提取步长（np.gradient 可接受一维坐标数组）
    # Xg[0, :] 是沿 x（经度）方向的一维数组
    # Yg[:, 0] 是沿 y（纬度）方向的一维数组
    dx_vec = Xg[0, :]   # shape (nx,)
    dy_vec = Yg[:, 0]   # shape (ny,)
    # 在 NaN 处用 0 填充以允许 gradient 计算，之后再还原 NaN
    nan_mask = np.isnan(Zg)
    Zg_filled = np.where(nan_mask, 0.0, Zg)
    # 计算界面梯度：?Z0/?y（沿行，即 axis=0）和 ?Z0/?x（沿列，即 axis=1）
    # np.gradient(f, y_vec, x_vec) 返回 [df/dy, df/dx]
    dZ_dy, dZ_dx = np.gradient(Zg_filled, dy_vec, dx_vec)
    # 计算法向量的模：‖n‖ = sqrt(1 + (?Z0/?x)? + (?Z0/?y)?)
    norm_n = np.sqrt(1.0 + dZ_dx ** 2 + dZ_dy ** 2)
    # g · n? = -g0 / ‖n‖（法向向上，重力向下，点积为负）
    gdot = -g0 / norm_n
    # 若需要翻转符号（向下法向或用户要求正值输出）
    if flip_sign:
        gdot = -gdot
    # 将原来 NaN 区域还原为 NaN，避免边界填充值污染结果
    gdot[nan_mask] = np.nan
    return gdot
def main():
    p = argparse.ArgumentParser(
        description=(
            "绘制重力加速度 g 与温跃层界面法向单位向量 n? 的点积 g·n? 的空间分布图。\n"
            "物理意义：g·n? 反映重力加速度在界面法方向上的分量（'法向重力分量'），\n"
            "与界面坡度直接相关，是斜压动力学中的重要驱动量。"
        )
    )
    p.add_argument(
        "--model",
        default="outputs/exp_6/model_final.pt",
        help="已训练模型路径，默认 outputs/exp_6/model_final.pt",
    )
    p.add_argument(
        "--train-csv",
        default="data/processed_data_mean_train.csv",
        help="训练集 CSV（用于拟合 scaler），默认 data/processed_data_mean_train.csv",
    )
    p.add_argument(
        "--test-csv",
        default="data/processed_data_mean_test.csv",
        help="测试集 CSV（提供散点经纬度等），默认 data/processed_data_mean_test.csv",
    )
    p.add_argument(
        "--outdir",
        default="outputs/exp_6",
        help="输出根目录，图片将保存到 <outdir>/fig/。默认 outputs/exp_6",
    )
    p.add_argument("--device", default="cpu", help="计算设备，例如 cpu 或 cuda:0")
    p.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="用于插值的散点最大数量；设为 0 使用全量（较慢）。默认 5000",
    )
    p.add_argument("--grid-nx", type=int, default=250, help="插值网格 x 方向分辨率，默认 250")
    p.add_argument("--grid-ny", type=int, default=250, help="插值网格 y 方向分辨率，默认 250")
    p.add_argument(
        "--interp",
        choices=["linear", "cubic", "nearest"],
        default="linear",
        help="散点→规则网格插值方法，默认 linear",
    )
    p.add_argument(
        "--g0",
        type=float,
        default=9.81,
        help="重力加速度标量（m/s?），默认 9.81",
    )
    p.add_argument(
        "--flip-sign",
        action="store_true",
        help=(
            "翻转点积符号：默认法向向上（z 分量 > 0），g·n? 为负值；"
            "指定此选项后等价于取向下法向，输出正值。"
        ),
    )
    p.add_argument(
        "--cmap",
        default="RdBu_r",
        help="matplotlib colormap，建议用发散色映射如 RdBu_r / seismic。默认 RdBu_r",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=64,
        help="contourf 颜色等级数；越大过渡越平滑。默认 64",
    )
    p.add_argument(
        "--title",
        default=None,
        help="图标题；默认根据 --flip-sign 自动生成",
    )
    args = p.parse_args()
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    fig_dir = outdir / "fig"
    save_path = fig_dir / "gdot_nhat_contourf.png"
    # ------------------------------------------------------------------
    # 1) 从训练集拟合 scaler（与训练保持一致）
    # ------------------------------------------------------------------
    print(f"[1/7] 从训练集拟合 scaler：{args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    _, _, _, _, _, _, scaler_mgr, _ = load_csv_data_from_df(train_df, device=device, fit_scaler=True)
    # ------------------------------------------------------------------
    # 2) 加载测试集并用 scaler 转换
    # ------------------------------------------------------------------
    print(f"[2/7] 加载测试集：{args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    t, x, y, z, _, _, _, _ = load_csv_data_from_df(test_df, device=device, scaler_mgr=scaler_mgr, fit_scaler=False)
    # ------------------------------------------------------------------
    # 3) 可选采样，避免点数过多时内存/速度问题
    # ------------------------------------------------------------------
    n = x.shape[0]
    if args.sample is not None and args.sample > 0 and n > args.sample:
        print(f"[3/7] 随机采样 {args.sample}/{n} 个散点")
        idx = torch.randperm(n, device=device)[: args.sample]
        t_s = t[idx]
        x_s = x[idx]
        y_s = y[idx]
        z_s = torch.zeros_like(x_s)
    else:
        print(f"[3/7] 使用全量 {n} 个散点")
        t_s, x_s, y_s = t, x, y
        z_s = torch.zeros_like(x)
    # ------------------------------------------------------------------
    # 4) 加载模型
    # ------------------------------------------------------------------
    print(f"[4/7] 加载模型：{args.model}")
    model = load_model(args.model, scaler_mgr, device)
    # ------------------------------------------------------------------
    # 5) 在 z=0 处前向推断，取 Z0（模型输出第 9 列，与 plot_thermocline_surface.py 一致）
    # ------------------------------------------------------------------
    print("[5/7] 推断 Z0 …")
    with torch.no_grad():
        out = model.forward(t_s, x_s, y_s, z_s)
        z0 = out[:, 9:10]
    # ------------------------------------------------------------------
    # 6) 反归一化至物理坐标
    # ------------------------------------------------------------------
    print("[6/7] 反归一化坐标 …")
    try:
        lon = scaler_mgr.inverse_x(x_s).reshape(-1)
        lat = scaler_mgr.inverse_y(y_s).reshape(-1)
        x_label = "Longitude (°E)"
        y_label = "Latitude (°N)"
    except Exception:
        lon = x_s.detach().cpu().numpy().reshape(-1)
        lat = y_s.detach().cpu().numpy().reshape(-1)
        x_label = "x (normalized)"
        y_label = "y (normalized)"
    try:
        z0_depth = scaler_mgr.inverse_transform_depth(z0).reshape(-1)
    except Exception:
        z0_depth = z0.detach().cpu().numpy().reshape(-1)
    # ------------------------------------------------------------------
    # 7) 插值到规则网格，计算梯度和 g·n?，绘图
    # ------------------------------------------------------------------
    print("[7/7] 插值→计算法向分量→绘图 …")
    # 7a) 把散点插值到规则经纬网格
    Xg, Yg, Zg = interpolate_to_grid(lon, lat, z0_depth, nx=args.grid_nx, ny=args.grid_ny, method=args.interp)
    # 7b) 计算 g · n?：
    #     - 构造法向量 n = (-?Z0/?x, -?Z0/?y, 1)（向上）
    #     - 归一化为单位向量 n? = n / ‖n‖
    #     - 点积 g · n? = (0,0,-g0) · n? = -g0·n?_z = -g0 / ‖n‖
    gdot = compute_gdot_nhat(Zg, Xg, Yg, g0=args.g0, flip_sign=args.flip_sign)
    # 7c) 设置标题和 colorbar 标签
    if args.title is not None:
        title = args.title
    else:
        if args.flip_sign:
            # 向下法向：点积为正值
            title = r"$\mathbf{g} \cdot \hat{\mathbf{n}}_{\downarrow}$ along thermocline normal (m/s?)"
        else:
            # 向上法向（默认）：点积为负值
            title = r"$\mathbf{g} \cdot \hat{\mathbf{n}}_{\uparrow}$ along thermocline normal (m/s?)"
    cbar_label = r"$g \cdot \hat{n}$  (m/s?)"
    # 7d) 绘制 2D contourf 并保存
    plot_contourf(
        Xg,
        Yg,
        gdot,
        str(save_path),
        title=title,
        xlabel=x_label,
        ylabel=y_label,
        cbar_label=cbar_label,
        cmap=args.cmap,
        levels=args.levels,
    )
    print(f"已保存图片：{save_path}")
if __name__ == "__main__":
    main()