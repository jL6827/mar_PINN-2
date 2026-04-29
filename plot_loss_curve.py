import pandas as pd
import matplotlib.pyplot as plt
import os

# ====================== 路径配置 ======================
loss_path = "outputs/exp_2/loss_history.csv"
save_dir = "outputs/exp_2/fig"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "loss_convergence_curves_zoomed_100.png")

# ====================== 读取数据 ======================
df = pd.read_csv(loss_path)

# ====================== 绘图配置 ======================
plt.figure(figsize=(9, 5))

plot_config = [
    ("total", "Total Loss", "#2c3e50"),
    ("data", "Data Loss", "#3498db"),
    ("phys", "PDE/Physics Loss", "#e74c3c"),
    ("geo", "Geometry Loss", "#2ecc71")
]

for col, label, color in plot_config:
    if col in df.columns:
        plt.plot(df["epoch"], df[col], label=label, color=color, linewidth=1.5, alpha=0.85)

# ====================== 关键：限定横轴范围为 0–100 epoch ======================
plt.xlim(left=0, right=100)

# ====================== 图表美化 ======================
plt.xlabel("Training Epoch", fontsize=14)
plt.ylabel("Loss Value", fontsize=14)
plt.title("Loss Convergence Curves (Zoomed on Early Stage)", fontsize=16, pad=15)
plt.legend(frameon=True, loc="upper right", fontsize=11)
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ 聚焦0-100 epoch的收敛曲线已保存到：{save_path}")