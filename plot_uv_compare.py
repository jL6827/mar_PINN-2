#!/usr/bin/env python3
"""
Plot predicted vs true comparisons for u and v from full_prediction.csv.

Generates:
- uv_scatter_compare.png  (2 subplots: u and v scatter with y=x line)
- uv_error_hist.png       (error histograms)
- uv_combined.png         (combined: scatter on top, hist on bottom)
- uv_metrics.txt          (MAE/RMSE/R2)

Usage:
  python plot_uv_compare.py --csv outputs/exp_9/full_prediction.csv --outdir outputs/exp_9/fig
  python plot_uv_compare.py --csv outputs/exp_6/test_pred/full_prediction.csv --outdir outputs/exp_6/test_pred/fig
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to full_prediction.csv")
    p.add_argument("--outdir", required=True, help="Directory to save figures")
    p.add_argument("--prefix", default="uv", help="Output filename prefix")
    p.add_argument("--sample", type=int, default=0, help="Randomly sample N points for scatter (0 = use all)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    required = ["u_true", "u_pred", "v_true", "v_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in CSV: {missing}. Available columns: {list(df.columns)}")

    if args.sample and args.sample > 0 and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=0).reset_index(drop=True)

    os.makedirs(args.outdir, exist_ok=True)

    u_true, u_pred = df["u_true"].values, df["u_pred"].values
    v_true, v_pred = df["v_true"].values, df["v_pred"].values

    u_mae, u_rmse, u_r2 = metrics(u_true, u_pred)
    v_mae, v_rmse, v_r2 = metrics(v_true, v_pred)

    # -------- scatter compare (original, kept unchanged) --------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    for ax, y_t, y_p, name, mae, rmse, r2 in [
        (axes[0], u_true, u_pred, "Zonal Velocity (u)", u_mae, u_rmse, u_r2),
        (axes[1], v_true, v_pred, "Meridional Velocity (v)", v_mae, v_rmse, v_r2),
    ]:
        ax.scatter(y_t, y_p, s=20, alpha=0.4)
        mn = min(np.min(y_t), np.min(y_p))
        mx = max(np.max(y_t), np.max(y_p))
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="1:1 line")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"{name}\nMAE={mae:.4g}, RMSE={rmse:.4g}, R²={r2:.4g}")
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out_scatter = os.path.join(args.outdir, f"{args.prefix}_scatter_compare.png")
    fig.savefig(out_scatter, bbox_inches="tight")
    plt.close(fig)

    # -------- error histogram (original, kept unchanged) --------
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    axes2[0].hist(u_pred - u_true, bins=60, alpha=0.85, color="#1f77b4")
    axes2[0].set_title("u Prediction Error Histogram (u_pred - u_true)")
    axes2[0].set_xlabel("Error")
    axes2[0].set_ylabel("Count")
    axes2[0].grid(alpha=0.3, ls=":")

    axes2[1].hist(v_pred - v_true, bins=60, alpha=0.85, color="#2ca02c")
    axes2[1].set_title("v Prediction Error Histogram (v_pred - v_true)")
    axes2[1].set_xlabel("Error")
    axes2[1].set_ylabel("Count")
    axes2[1].grid(alpha=0.3, ls=":")

    fig2.tight_layout()
    out_hist = os.path.join(args.outdir, f"{args.prefix}_error_hist.png")
    fig2.savefig(out_hist, bbox_inches="tight")
    plt.close(fig2)

    # -------- NEW: Combined figure (scatter top, hist bottom) --------
        # -------- NEW: Combined figure (scatter top, hist bottom) --------
    fig_combined = plt.figure(figsize=(10, 8), dpi=300)
    
    # Create 2x2 grid
    gs = fig_combined.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    
    # Top row: scatter plots
    ax_comb_scatter_u = fig_combined.add_subplot(gs[0, 0])
    ax_comb_scatter_v = fig_combined.add_subplot(gs[0, 1])
    
    # Bottom row: histograms
    ax_comb_hist_u = fig_combined.add_subplot(gs[1, 0])
    ax_comb_hist_v = fig_combined.add_subplot(gs[1, 1])
    
    # ---- Plot scatter u (left top) ----
    ax_comb_scatter_u.scatter(u_true, u_pred, s=20, alpha=0.4)
    mn_u = min(np.min(u_true), np.min(u_pred))
    mx_u = max(np.max(u_true), np.max(u_pred))
    ax_comb_scatter_u.plot([mn_u, mx_u], [mn_u, mx_u], "r--", lw=1.5, label="1:1 line")
    ax_comb_scatter_u.set_xlabel("True Value")
    ax_comb_scatter_u.set_ylabel("Predicted Value")
    ax_comb_scatter_u.set_title(f"Zonal Velocity (u)\nMAE={u_mae:.4g}, RMSE={u_rmse:.4g}, R²={u_r2:.4g}")
    ax_comb_scatter_u.grid(alpha=0.3, ls=":")
    ax_comb_scatter_u.legend(fontsize=8, loc="upper left")
    ax_comb_scatter_u.tick_params(axis='x', rotation=30)  # 倾斜横轴刻度
    
    # ---- Plot scatter v (right top) ----
    ax_comb_scatter_v.scatter(v_true, v_pred, s=20, alpha=0.4)
    mn_v = min(np.min(v_true), np.min(v_pred))
    mx_v = max(np.max(v_true), np.max(v_pred))
    ax_comb_scatter_v.plot([mn_v, mx_v], [mn_v, mx_v], "r--", lw=1.5, label="1:1 line")
    ax_comb_scatter_v.set_xlabel("True Value")
    ax_comb_scatter_v.set_ylabel("Predicted Value")
    ax_comb_scatter_v.set_title(f"Meridional Velocity (v)\nMAE={v_mae:.4g}, RMSE={v_rmse:.4g}, R²={v_r2:.4g}")
    ax_comb_scatter_v.grid(alpha=0.3, ls=":")
    ax_comb_scatter_v.legend(fontsize=8, loc="upper left")
    ax_comb_scatter_v.tick_params(axis='x', rotation=30)  # 倾斜横轴刻度
    
    # ---- Plot histogram u (left bottom) ----
    ax_comb_hist_u.hist(u_pred - u_true, bins=60, alpha=0.85, color="#1f77b4")
    ax_comb_hist_u.set_title("u Prediction Error Histogram")
    ax_comb_hist_u.set_xlabel("Error (u_pred - u_true)")
    ax_comb_hist_u.set_ylabel("Count")
    ax_comb_hist_u.grid(alpha=0.3, ls=":")
    ax_comb_hist_u.tick_params(axis='x', rotation=30)  # 倾斜横轴刻度
    
    # ---- Plot histogram v (right bottom) ----
    ax_comb_hist_v.hist(v_pred - v_true, bins=60, alpha=0.85, color="#2ca02c")
    ax_comb_hist_v.set_title("v Prediction Error Histogram")
    ax_comb_hist_v.set_xlabel("Error (v_pred - v_true)")
    ax_comb_hist_v.set_ylabel("Count")
    ax_comb_hist_v.grid(alpha=0.3, ls=":")
    ax_comb_hist_v.tick_params(axis='x', rotation=30)  # 倾斜横轴刻度
    
    # Add overall title
    fig_combined.suptitle("Velocity Prediction: u & v", fontsize=14, fontweight="bold", y=0.98)
    
    fig_combined.tight_layout()
    out_combined = os.path.join(args.outdir, f"{args.prefix}_combined.png")
    fig_combined.savefig(out_combined, bbox_inches="tight", dpi=300)
    plt.close(fig_combined)

    # -------- metrics txt (original, kept unchanged) --------
    out_txt = os.path.join(args.outdir, f"{args.prefix}_metrics.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Velocity prediction metrics\n")
        f.write(f"Source CSV: {args.csv}\n\n")
        f.write("Zonal Velocity (u)\n")
        f.write(f"  MAE : {u_mae:.8f}\n")
        f.write(f"  RMSE: {u_rmse:.8f}\n")
        f.write(f"  R2  : {u_r2:.8f}\n\n")
        f.write("Meridional Velocity (v)\n")
        f.write(f"  MAE : {v_mae:.8f}\n")
        f.write(f"  RMSE: {v_rmse:.8f}\n")
        f.write(f"  R2  : {v_r2:.8f}\n")

    print("✅ Saved:")
    print(" -", out_scatter)
    print(" -", out_hist)
    print(" -", out_combined)
    print(" -", out_txt)


if __name__ == "__main__":
    main()