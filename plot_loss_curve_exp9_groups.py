#!/usr/bin/env python3
"""
Plot grouped convergence curves from loss_history.csv (exp9-style grouped figures).

Produces:
  1) 1x3 subplots: phys, cont, geo
  2) 1x3 subplots: b0_reg, b1_reg, Ro
  3) 1x2 subplots: data, dir
Notes:
  - Column 'act' is never plotted.
  - Last epoch value is marked with a red dashed horizontal line, 
    with the value labeled inside the plot frame near the right boundary
    with a semi-transparent white background, using Arial font.

Example:
  python plot_loss_curve_exp9_groups.py --csv outputs/exp_9/loss_history.csv --outdir outputs/exp_9/fig --max-epoch 100 --smooth 5 --logy
"""

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "total": "Total Loss",
    "phys": "PDE Residual",
    "data": "Data Fitting Loss",
    "geo": "Geometric Constraint",
    "cont": "Continuity Residual",
    "dir": "Direction Loss",
    "act": "Activation Regularization",
    "z0_reg": "Z\u2080 Regularization",
    "ro_reg": "Rossby Number (Ro) Regularization",
    "b0_reg": "B0 Regularization",
    "b1_reg": "B1 Regularization",
    "ro_change_penalty": "Ro Change Penalty",
    "lr_other": "Learning Rate (Other Parameters)",
    "lr_Ro": "Learning Rate (Ro Parameter)",
    "Ro": "Rossby Number (Ro)",
}


def label_of(col: str) -> str:
    return LABELS.get(col, col)


def rolling(s: pd.Series, w: int) -> pd.Series:
    if w is None or w <= 1:
        return s
    return s.rolling(window=w, min_periods=1, center=False).mean()


def plot_panel(df_sel, x, cols, out_path, title, *, nrows, ncols, smooth, logy, dpi, max_epoch):
    cols = [c for c in cols if c != "act"]  # hard-exclude act

    present = [c for c in cols if c in df_sel.columns]
    missing = [c for c in cols if c not in df_sel.columns]
    if missing:
        print(f"⚠️ Missing columns skipped in {os.path.basename(out_path)}: {missing}")
    if not present:
        print(f"⚠️ Plot skipped (no columns present): {out_path}")
        return

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.0, nrows * 3.6),
        dpi=dpi,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    # 适中的红色
    medium_red = '#CC3333'

    for i, col in enumerate(present):
        ax = axes_flat[i]
        y = rolling(df_sel[col], smooth)

        if logy:
            ax.semilogy(x, y, lw=1.4)
        else:
            ax.plot(x, y, lw=1.4)

        # 获取最后一个epoch的值
        last_val = df_sel[col].iloc[-1]

        # 绘制红色水平虚线
        ax.axhline(y=last_val, color=medium_red, linestyle='--', linewidth=0.8, alpha=0.7)

        # 数值标注在图内部右边界的左侧，使用Arial字体
        ax.text(0.95, last_val, f'{last_val:.4g}',
                transform=ax.get_yaxis_transform(),
                ha='right',
                va='bottom',
                color=medium_red,
                fontweight='bold',
                fontsize=12,
                fontfamily='Arial',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor='none',
                         alpha=0.85))

        ax.set_title(label_of(col), fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.set_xlim(0, max_epoch)
        ax.grid(True, ls=":", alpha=0.6)

    for j in range(len(present), nrows * ncols):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Split loss plots into grouped panels (exp9 v3).")
    p.add_argument("--csv", required=True, help="Path to loss_history.csv")
    p.add_argument("--outdir", required=True, help="Output directory to save figures")
    p.add_argument("--max-epoch", type=int, default=10000, help="Max epoch to include on x axis (inclusive)")
    p.add_argument("--smooth", type=int, default=1, help="Rolling mean window. Default 1 (no smoothing).")
    p.add_argument("--logy", action="store_true", help="Use log scale on y axis for panels")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI for saved PNGs")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "epoch" not in df.columns:
        raise RuntimeError("CSV must contain 'epoch' column")

    df_sel = df[df["epoch"].between(0, args.max_epoch)].copy()
    if df_sel.empty:
        raise RuntimeError(f"No rows with epoch in 0..{args.max_epoch} in {args.csv}")

    os.makedirs(args.outdir, exist_ok=True)
    x = df_sel["epoch"].values

    # (A) phys / cont / geo -> 1x3
    out_pcg = os.path.join(args.outdir, f"panel_phys_cont_geo_1x3_epoch0_{args.max_epoch}.png")
    plot_panel(
        df_sel, x,
        cols=["phys", "cont", "geo"],
        out_path=out_pcg,
        title=f"PDE / Continuity / Geometry (Epoch 0–{args.max_epoch})",
        nrows=1, ncols=3,
        smooth=args.smooth, logy=args.logy, dpi=args.dpi, max_epoch=args.max_epoch,
    )

    # (B) B0 / B1 / Ro -> 1x3
    out_b = os.path.join(args.outdir, f"panel_B0_B1_Ro_1x3_epoch0_{args.max_epoch}.png")
    plot_panel(
        df_sel, x,
        cols=["b0_reg", "b1_reg", "Ro"],
        out_path=out_b,
        title=f"B0 / B1 / Ro (Epoch 0–{args.max_epoch})",
        nrows=1, ncols=3,
        smooth=args.smooth, logy=args.logy, dpi=args.dpi, max_epoch=args.max_epoch,
    )

    # (C) Data / Dir -> 1x2
    out_dd = os.path.join(args.outdir, f"panel_data_dir_1x2_epoch0_{args.max_epoch}.png")
    plot_panel(
        df_sel, x,
        cols=["data", "dir"],
        out_path=out_dd,
        title=f"Data Fitting & Direction Loss (Epoch 0–{args.max_epoch})",
        nrows=1, ncols=2,
        smooth=args.smooth, logy=args.logy, dpi=args.dpi, max_epoch=args.max_epoch,
    )


if __name__ == "__main__":
    main()