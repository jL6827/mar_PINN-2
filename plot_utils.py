import matplotlib.pyplot as plt
import numpy as np
import os

def plot_and_save(u_pred, v_pred, u_true, v_true, output_dir):
    u_pred = u_pred.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    u_true = u_true.detach().cpu().numpy()
    v_true = v_true.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.quiver(u_true, v_true, color='blue')
    plt.title('True Velocity')
    plt.subplot(1, 2, 2)
    plt.quiver(u_pred, v_pred, color='red')
    plt.title('Predicted Velocity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_comparison.png'))
    plt.close()

    # 统计分析
    stats = {
        'u_pred_mean': np.mean(u_pred),
        'v_pred_mean': np.mean(v_pred),
        'u_true_mean': np.mean(u_true),
        'v_true_mean': np.mean(v_true),
        'u_error_rmse': np.sqrt(np.mean((u_pred - u_true) ** 2)),
        'v_error_rmse': np.sqrt(np.mean((v_pred - v_true) ** 2))
    }

    with open(os.path.join(output_dir, 'prediction_stats.txt'), 'w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.6f}\n")
