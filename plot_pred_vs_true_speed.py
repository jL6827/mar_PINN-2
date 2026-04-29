#!/usr/bin/env python3
# plot_pred_vs_true_speed.py
# 说明：读取 evaluate.py 生成的 velocity_comparison_data.csv，绘制u分量、v分量的预测vs真实对比散点图，
#      并在图中标注 R^2、RMSE 等指标。
#
# 用法示例：
#   python plot_pred_vs_true_speed.py --outdir outputs/exp_1
#   python plot_pred_vs_true_speed.py --csv outputs/exp_1/velocity_comparison_data.csv
#
# 输出：
#   <outdir>/fig/u_pred_vs_true_scatter.png
#   <outdir>/fig/v_pred_vs_true_scatter.png
#   (可选) <outdir>/fig/speed_pred_vs_true_scatter.png

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_r2_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    """返回 (r2, rmse)。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mse = np.mean((y_pred - y_true) ** 2)
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0.0 else float("nan")
    return r2, rmse


def plot_scatter_pred_vs_true(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    save_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
):
    """通用的散点图绘制函数"""
    true_vals = np.asarray(true_vals, dtype=float).ravel()
    pred_vals = np.asarray(pred_vals, dtype=float).ravel()
    if true_vals.shape != pred_vals.shape:
        raise ValueError(f"shape 不一致：true={true_vals.shape}, pred={pred_vals.shape}")

    r2, rmse = compute_r2_rmse(true_vals, pred_vals)

    vmin = float(np.nanmin([true_vals.min(), pred_vals.min()]))
    vmax = float(np.nanmax([true_vals.max(), pred_vals.max()]))

    plt.figure(figsize=(6.2, 6.2), dpi=150)
    plt.scatter(true_vals, pred_vals, s=10, alpha=0.6, edgecolors="none")

    # 理想线 y=x
    plt.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.6, label="Ideal: y=x")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    txt = f"$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}"
    plt.text(
        0.05,
        0.95,
        txt,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )

    plt.grid(True, ls=":", lw=0.7, alpha=0.7)
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.axis("equal")
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(
        description="绘制u/v分量的预测 vs 真实值散点图（基于 velocity_comparison_data.csv），并标注 R^2、RMSE。"
    )
    p.add_argument(
        "--csv",
        default=None,
        help="velocity_comparison_data.csv 的路径。若不提供，可用 --outdir 自动推断。",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="evaluate.py 的输出目录，例如 outputs/exp_1。若提供，会默认读取 <outdir>/velocity_comparison_data.csv，并把图保存到 <outdir>/fig/",
    )
    p.add_argument(
        "--plot-speed",
        action="store_true",
        help="是否额外绘制速度大小（合速度）的对比图",
    )
    args = p.parse_args()

    if args.csv is None and args.outdir is None:
        raise SystemExit("请提供 --csv 或 --outdir 至少一个参数")

    # 确定CSV路径和输出目录
    if args.csv is None:
        csv_path = Path(args.outdir) / "velocity_comparison_data.csv"
    else:
        csv_path = Path(args.csv)

    if args.outdir is None:
        outdir = csv_path.parent
    else:
        outdir = Path(args.outdir)

    # 检查CSV文件是否存在
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 comparison CSV：{csv_path}")

    # 读取数据并验证列
    df = pd.read_csv(csv_path)
    required = {"u_true", "u_pred", "v_true", "v_pred"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV 缺少必要列：{required}，当前列：{list(df.columns)}")

    # 提取u、v分量数据
    u_true = df["u_true"].to_numpy(dtype=float)
    u_pred = df["u_pred"].to_numpy(dtype=float)
    v_true = df["v_true"].to_numpy(dtype=float)
    v_pred = df["v_pred"].to_numpy(dtype=float)

    # 定义输出目录
    fig_dir = outdir / "fig"

    # 绘制u分量对比图
    u_save_path = fig_dir / "u_pred_vs_true_scatter.png"
    plot_scatter_pred_vs_true(
        true_vals=u_true,
        pred_vals=u_pred,
        save_path=str(u_save_path),
        title="Predicted vs True U-component",
        xlabel="True U-component",
        ylabel="Predicted U-component"
    )
    print(f"已保存u分量散点图: {u_save_path}")

    # 绘制v分量对比图
    v_save_path = fig_dir / "v_pred_vs_true_scatter.png"
    plot_scatter_pred_vs_true(
        true_vals=v_true,
        pred_vals=v_pred,
        save_path=str(v_save_path),
        title="Predicted vs True V-component",
        xlabel="True V-component",
        ylabel="Predicted V-component"
    )
    print(f"已保存v分量散点图: {v_save_path}")

    # 可选：绘制速度大小对比图
    if args.plot_speed:
        speed_true = np.sqrt(u_true ** 2 + v_true ** 2)
        speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        speed_save_path = fig_dir / "speed_pred_vs_true_scatter.png"
        plot_scatter_pred_vs_true(
            true_vals=speed_true,
            pred_vals=speed_pred,
            save_path=str(speed_save_path),
            title="Predicted vs True Speed",
            xlabel="True speed (observed)",
            ylabel="Predicted speed (model)"
        )
        print(f"已保存速度大小散点图: {speed_save_path}")


if __name__ == "__main__":
    main()