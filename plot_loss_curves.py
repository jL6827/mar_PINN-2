#!/usr/bin/env python3
"""
Plot per-component training loss curves (each variable in its own subplot)
Default epoch range: 0..100

Usage:
    python plot_loss_curves.py --csv outputs/exp_2/loss_history.csv --outdir outputs/exp_2 --max-epoch 100
"""
import argparse
import os
from math import ceil, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 如果在无显示环境下运行，仍能保存图片


def safe_plot(ax, x, y, label, logy=False):
    if logy:
        ax.semilogy(x, y, label=label)
    else:
        ax.plot(x, y, label=label)
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize="small")


def main():
    p = argparse.ArgumentParser(description="Plot per-component loss curves from loss_history.csv")
    p.add_argument("--csv", required=True, help="Path to loss_history.csv")
    p.add_argument("--outdir", required=True, help="Output directory to save figure")
    p.add_argument("--max-epoch", type=int, default=100, help="Max epoch to plot (inclusive). Default 100")
    p.add_argument(
        "--vars",
        nargs="+",
        default=["total", "data", "phys", "geo", "cont", "dir", "act", "z0_reg"],
        help="Loss columns to plot (default common components).",
    )
    p.add_argument("--logy", action="store_true", help="Use log scale for y-axis")
    p.add_argument("--dpi", type=int, default=150, help="Output figure DPI")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "epoch" not in df.columns:
        raise ValueError("CSV must contain an 'epoch' column")

    # filter by epoch range
    df_sel = df[df["epoch"].between(0, args.max_epoch)].copy()
    if df_sel.empty:
        raise ValueError(f"No rows in CSV with epoch in 0..{args.max_epoch}")

    # determine which columns actually exist
    vars_to_plot = [v for v in args.vars if v in df_sel.columns]
    missing = [v for v in args.vars if v not in df_sel.columns]
    if missing:
        print(f"⚠️ 这些列在 CSV 中未找到，将被跳过: {missing}")

    n = len(vars_to_plot)
    if n == 0:
        raise ValueError("没有可绘制的列，请检查 --vars 参数或 CSV 内容")

    # layout: try near-square grid
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / cols))

    fig_w = max(6, cols * 4)
    fig_h = max(3, rows * 3)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=args.dpi, squeeze=False)
    axes_flat = axes.flatten()

    x = df_sel["epoch"].values

    # autoscale y-limits later
    y_mins = []
    y_maxs = []

    for i, var in enumerate(vars_to_plot):
        ax = axes_flat[i]
        y = df_sel[var].values
        # If column is constant or nearly zero, plot raw values
        safe_plot(ax, x, y, label=var, logy=args.logy)
        ax.set_xlabel("epoch")
        ax.set_ylabel(var)
        ax.set_title(var)
        # record for autoscale
        finite = y[np.isfinite(y)]
        if finite.size:
            y_mins.append(finite.min())
            y_maxs.append(finite.max())

    # hide unused subplots
    for j in range(n, rows * cols):
        axes_flat[j].axis("off")

    plt.tight_layout()

    # save
    os.makedirs(args.outdir, exist_ok=True)
    out_png = os.path.join(args.outdir, f"loss_components_epoch0_{args.max_epoch}.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 已保存损失分量图（0..{args.max_epoch} epoch）到: {out_png}")

    # also save a compact combined plot (multiple curves in one figure)
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=args.dpi)
    for var in vars_to_plot:
        y = df_sel[var].values
        if args.logy:
            ax2.semilogy(x, y, label=var)
        else:
            ax2.plot(x, y, label=var)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title(f"Loss components (epoch 0..{args.max_epoch})")
    ax2.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax2.legend(fontsize="small", ncol=2)
    out_png2 = os.path.join(args.outdir, f"loss_components_combined_epoch0_{args.max_epoch}.png")
    fig2.tight_layout()
    fig2.savefig(out_png2, bbox_inches="tight")
    plt.close(fig2)
    print(f"✅ 已保存合并损失图到: {out_png2}")


if __name__ == "__main__":
    main()
