#!/usr/bin/env python3
"""
Plot loss convergence curves from loss_history.csv (paper-ready labels).

Produces:
 - Focus plot with Total / PDE / Data / Geometry (epoch 0..max_epoch)
 - Combined plot for selected variables
 - Individual subplots for numeric columns (excluding lr* by default)

Usage:
    python plot_loss_curve.py --csv outputs/exp_7/loss_history.csv --outdir outputs/exp_7/fig --max-epoch 100 --smooth 5
"""
import argparse
import os
from math import ceil, sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Column name -> paper-ready label
LABELS = {
    "total": "Total Loss",
    "phys": "PDE Residual Loss",
    "data": "Data Fitting Loss",
    "geo": "Geometric Constraint Loss",
    "cont": "Continuity Residual Loss",
    "dir": "Direction Loss",
    "act": "Activation Regularization Loss",
    "z0_reg": "Z\u2080 Regularization Loss",
    "ro_reg": "Rossby Number (Ro) Regularization Loss",
    "b0_reg": "B0 Regularization Loss",
    "b1_reg": "B1 Regularization Loss",
    "ro_change_penalty": "Ro Change Penalty",
    "lr_other": "Learning Rate (Other Parameters)",
    "lr_Ro": "Learning Rate (Ro Parameter)",
    "Ro": "Rossby Number (Ro)",
}


def label_of(col: str) -> str:
    return LABELS.get(col, col)  # fallback: show raw column name


def rolling(s: pd.Series, w: int) -> pd.Series:
    if w is None or w <= 1:
        return s
    return s.rolling(window=w, min_periods=1, center=False).mean()


def main():
    p = argparse.ArgumentParser(description="Plot loss convergence curves (paper-ready labels)")
    p.add_argument("--csv", required=True, help="Path to loss_history.csv")
    p.add_argument("--outdir", required=True, help="Output directory to save figures")
    p.add_argument("--max-epoch", type=int, default=100, help="Max epoch to include on x axis (inclusive)")
    p.add_argument("--smooth", type=int, default=1, help="Rolling mean window. Default 1 (no smoothing).")
    p.add_argument("--logy", action="store_true", help="Use log scale on y axis in focus/combined plots")
    p.add_argument(
        "--vars",
        nargs="+",
        default=["total", "data", "phys", "geo", "cont", "dir", "act", "z0_reg"],
        help="Columns to include in combined plot (defaults are common loss components)",
    )
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

    # ------------------ Focus plot: Total / PDE / Data / Geometry ------------------
    focus_cols = ["total", "phys", "data", "geo"]
    focus_present = [c for c in focus_cols if c in df_sel.columns]
    if focus_present:
        plt.figure(figsize=(8.5, 5.2))
        for col in focus_present:
            y = rolling(df_sel[col], args.smooth)
            if args.logy:
                plt.semilogy(x, y, label=label_of(col), linewidth=1.8, alpha=0.92)
            else:
                plt.plot(x, y, label=label_of(col), linewidth=1.8, alpha=0.92)

        plt.xlim(0, args.max_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title(f"Loss Convergence (Epoch 0–{args.max_epoch})")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(frameon=True, fontsize=10, loc="upper right")
        out_focus = os.path.join(args.outdir, f"loss_focus_epoch0_{args.max_epoch}.png")
        plt.tight_layout()
        plt.savefig(out_focus, dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"✅ Focus plot saved: {out_focus}")
    else:
        print("⚠️ Focus plot skipped: none of columns total/phys/data/geo found.")

    # ------------------ Combined plot for chosen vars ------------------
    vars_present = [v for v in args.vars if v in df_sel.columns]
    missing = [v for v in args.vars if v not in df_sel.columns]
    if missing:
        print(f"⚠️ Missing columns skipped in combined plot: {missing}")

    if vars_present:
        plt.figure(figsize=(10, 5.3))
        for col in vars_present:
            y = rolling(df_sel[col], args.smooth)
            if args.logy:
                plt.semilogy(x, y, label=label_of(col), linewidth=1.25, alpha=0.9)
            else:
                plt.plot(x, y, label=label_of(col), linewidth=1.25, alpha=0.9)

        plt.xlim(0, args.max_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"Selected Loss Components (Epoch 0–{args.max_epoch})")
        plt.grid(alpha=0.3, linestyle=":")
        plt.legend(fontsize=9, ncol=2, frameon=True)
        out_combined = os.path.join(args.outdir, f"loss_selected_combined_epoch0_{args.max_epoch}.png")
        plt.tight_layout()
        plt.savefig(out_combined, dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"✅ Combined plot saved: {out_combined}")
    else:
        print("⚠️ Combined plot skipped: no requested vars exist in CSV.")

    # ------------------ Individual subplots for numeric columns ------------------
    # Exclude learning-rate curves by default from the grid (paper figures usually don't mix them with losses).
    exclude_prefixes = ("lr",)
    numeric_cols = [
        c for c in df_sel.columns
        if c != "epoch"
        and pd.api.types.is_numeric_dtype(df_sel[c])
        and not c.startswith(exclude_prefixes)
    ]
    # Also exclude Ro by default (unless user explicitly requested it in --vars)
    if "Ro" in numeric_cols and "Ro" not in args.vars:
        numeric_cols.remove("Ro")

    if numeric_cols:
        n = len(numeric_cols)
        grid_cols = int(ceil(sqrt(n)))
        grid_rows = int(ceil(n / grid_cols))

        fig_w = max(7, grid_cols * 4.2)
        fig_h = max(4, grid_rows * 2.8)
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_w, fig_h), dpi=args.dpi, squeeze=False)
        axes_flat = axes.flatten()

        for i, col in enumerate(numeric_cols):
            ax = axes_flat[i]
            y = rolling(df_sel[col], args.smooth)

            finite = np.asarray(y)[np.isfinite(y)]
            use_log = False
            if finite.size:
                vmin, vmax = float(finite.min()), float(finite.max())
                if vmin > 0 and vmax / max(vmin, 1e-12) > 1e3:
                    use_log = True

            if use_log:
                ax.semilogy(x, y, lw=1.2)
            else:
                ax.plot(x, y, lw=1.2)

            ax.set_title(label_of(col), fontsize=10)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel("Value", fontsize=9)
            ax.grid(True, ls=":", alpha=0.6)

        for j in range(n, grid_rows * grid_cols):
            axes_flat[j].axis("off")

        plt.tight_layout()
        out_ind = os.path.join(args.outdir, f"loss_individual_epoch0_{args.max_epoch}.png")
        fig.savefig(out_ind, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Individual subplots saved: {out_ind}")
    else:
        print("⚠️ Individual subplots skipped: no numeric columns found.")


if __name__ == "__main__":
    main()