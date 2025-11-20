import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from config import ModelConfig
from scaler_manager import ScalerManager

def save_model(model, output_dir, filename="trained_model.pt"):
    model_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"✅ 模型已保存到: {model_path}")
#可视化报告保存
def save_visual_report(x, y, u_pred, v_pred, output_dir, Z0=None, t=None):
    # 转为 numpy
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    u_np = u_pred.detach().cpu().numpy().flatten()
    v_np = v_pred.detach().cpu().numpy().flatten()

    # ✅ 保存速度场图（quiver）
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(x_np, y_np, u_np, v_np, scale=50, width=0.002, color='blue')
    ax.set_title("Velocity Field (u, v)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis('equal')
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "velocity_field.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"🖼️ 速度场图已保存到: {fig_path}")

    # ✅ 保存 CSV
    data = {
        'x': x_np,
        'y': y_np,
        'u_pred': u_np,
        'v_pred': v_np
    }
    if Z0 is not None:
        data['Z0'] = Z0.detach().cpu().numpy().flatten()
    if t is not None:
        data['t'] = t.detach().cpu().numpy().flatten()

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "velocity_field.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 速度场数据已保存到: {csv_path}")


#模型参数保存
class PredictionSaver:
    def __init__(self, model, scaler_mgr, output_dir):
        self.model = model
        self.scaler_mgr = scaler_mgr
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def inverse_all(self, x, y, z, t, u_pred, v_pred, u_true, v_true, Z0):
        return {
            'x': self.scaler_mgr.inverse_x(x).reshape(-1),
            'y': self.scaler_mgr.inverse_y(y).reshape(-1),
            'z': self.scaler_mgr.inverse_transform_depth(z).reshape(-1),
            't': self.scaler_mgr.inverse_time(t).reshape(-1),
            'u_pred': self.scaler_mgr.inverse_velocity(u_pred).reshape(-1),
            'v_pred': self.scaler_mgr.inverse_velocity(v_pred).reshape(-1),
            'u_true': self.scaler_mgr.inverse_velocity(u_true).reshape(-1) if u_true is not None else None,
            'v_true': self.scaler_mgr.inverse_velocity(v_true).reshape(-1) if v_true is not None else None,
            'Z0': self.scaler_mgr.inverse_transform_depth(Z0).reshape(-1)
        }

    def save_Z0(self, x, y, z=None, t=None, u_pred=None, v_pred=None, Z0=None):
        if Z0 is None:
            raise ValueError("Z0 must be provided.")
        data = self.inverse_all(x, y, z, t, u_pred, v_pred, None, None, Z0)
        df = pd.DataFrame({k: v for k, v in data.items() if v is not None})
        path = os.path.join(self.output_dir, "Z0_prediction.csv")
        df.to_csv(path, index=False)
        print(f"✅ Z₀ 已保存到: {path}")

    def save_full_prediction(self, original_df, u_pred, v_pred, u_true, v_true, Z0, x, y, z, t):
        import datetime

        self.model.eval()
        with torch.no_grad():
            data = self.inverse_all(x, y, z, t, u_pred, v_pred, u_true, v_true, Z0)

        df = pd.DataFrame(data)

        # ✅ 合并原始数据列
        for col in ['time', 'segment_id', 'latitude', 'longitude', 'depth', 'so', 'thetao', 'uo', 'vo']:
            if col in original_df.columns:
                df[col] = original_df[col].values[:len(df)]

        # ✅ 添加时间字符串列（假设起始时间为 2023-03-01 00:00:00）
        t0 = datetime.datetime(2023, 3, 1, 0, 0, 0)
        if 'time' in df.columns:
            df['date_str'] = df['time'].apply(
                lambda s: (t0 + datetime.timedelta(seconds=s)).strftime('%Y-%m-%d %H:%M:%S'))

        # ✅ 保存 CSV
        path = os.path.join(self.output_dir, "full_prediction.csv")
        df.to_csv(path, index=False)
        print(f"✅ 完整预测结果已保存到: {path}")

    def compute_direction_metrics(self, u_pred, v_pred, u_true, v_true, g1, g2, Z0, x, y):
        x.requires_grad_(True)
        y.requires_grad_(True)
        Z0_x = torch.autograd.grad(Z0, x, grad_outputs=torch.ones_like(Z0), create_graph=True)[0]
        Z0_y = torch.autograd.grad(Z0, y, grad_outputs=torch.ones_like(Z0), create_graph=True)[0]

        g_vec = torch.stack([g1.flatten(), g2.flatten()], dim=1)
        Z0_grad = torch.stack([Z0_x.flatten(), Z0_y.flatten()], dim=1)
        g_unit = g_vec / (g_vec.norm(dim=1, keepdim=True) + 1e-8)
        Z0_unit = Z0_grad / (Z0_grad.norm(dim=1, keepdim=True) + 1e-8)
        dot_gz = (g_unit * Z0_unit).sum(dim=1)

        u_vec_true = torch.stack([u_true.flatten(), v_true.flatten()], dim=1)
        u_vec_pred = torch.stack([u_pred.flatten(), v_pred.flatten()], dim=1)
        u_unit_true = u_vec_true / (u_vec_true.norm(dim=1, keepdim=True) + 1e-8)
        u_unit_pred = u_vec_pred / (u_vec_pred.norm(dim=1, keepdim=True) + 1e-8)
        dot_uv = (u_unit_true * u_unit_pred).sum(dim=1)

        return dot_gz, dot_uv

    def save_direction_consistency(self, u_pred, v_pred, u_true, v_true, dot_gz, dot_uv):
        path = os.path.join(self.output_dir, "direction_consistency.txt")
        with open(path, "w") as f:
            f.write("📊 速度预测误差:\n")
            f.write(f"u MAE: {(u_pred - u_true).abs().mean().item():.6f}\n")
            f.write(f"v MAE: {(v_pred - v_true).abs().mean().item():.6f}\n\n")
            f.write(" g · ∇Z₀:\n")
            f.write(f"mean: {dot_gz.mean().item():.6f}, std: {dot_gz.std().item():.6f}\n\n")
            f.write(" u_pred vs u_true:\n")
            f.write(f"mean: {dot_uv.mean().item():.6f}, std: {dot_uv.std().item():.6f}\n")
        print(f"方向一致性报告已保存到: {path}")

    def save_diagnostics(self, final_losses, g_dot_gradZ0=None, cos_sim=None, extra_metrics=None):
        path = os.path.join(self.output_dir, "diagnostic_metrics.txt")
        with open(path, "w") as f:
            f.write("📌 模型参数:\n")
            f.write(f"Ro: {self.model.Ro.item():.6f}\n")
            f.write(f"B0_scalar: {self.model.B0_scalar.item():.6f}\n")
            f.write(f"B1_scalar: {self.model.B1_scalar.item():.6f}\n\n")
            if g_dot_gradZ0 is not None:
                f.write("∇g · ∇Z₀:\n")
                f.write(f"mean: {g_dot_gradZ0.mean().item():.6f}, std: {g_dot_gradZ0.std().item():.6f}\n\n")
            if cos_sim is not None:
                f.write("📐 Predicted vs True Velocity Direction:\n")
                f.write(f"mean: {cos_sim.mean().item():.6f}, std: {cos_sim.std().item():.6f}\n\n")
            f.write("📉 损失项:\n")
            for name, value in final_losses.items():
                f.write(f"{name}: {value.item():.6f}\n")
            if extra_metrics:
                f.write("\n📊 其他指标:\n")
                for name, value in extra_metrics.items():
                    f.write(f"{name}: {value:.6f}\n")
        print(f"✅ 诊断指标已保存到: {path}")

    def compute_cos_similarity(self, u_pred, v_pred, u_true, v_true):
        u_vec_true = torch.stack([u_true.flatten(), v_true.flatten()], dim=1)
        u_vec_pred = torch.stack([u_pred.flatten(), v_pred.flatten()], dim=1)
        u_unit_true = u_vec_true / (u_vec_true.norm(dim=1, keepdim=True) + 1e-8)
        u_unit_pred = u_vec_pred / (u_vec_pred.norm(dim=1, keepdim=True) + 1e-8)
        return (u_unit_true * u_unit_pred).sum(dim=1)

    def compute_g_dot_gradZ0(self, g1, g2, Z0, x, y):
        x.requires_grad_(True)
        y.requires_grad_(True)
        Z0_x = torch.autograd.grad(Z0, x, grad_outputs=torch.ones_like(Z0), create_graph=True)[0]
        Z0_y = torch.autograd.grad(Z0, y, grad_outputs=torch.ones_like(Z0), create_graph=True)[0]
        g_vec = torch.stack([g1.flatten(), g2.flatten()], dim=1)
        Z0_grad = torch.stack([Z0_x.flatten(), Z0_y.flatten()], dim=1)
        g_unit = g_vec / (g_vec.norm(dim=1, keepdim=True) + 1e-8)
        Z0_unit = Z0_grad / (Z0_grad.norm(dim=1, keepdim=True) + 1e-8)
        return (g_unit * Z0_unit).sum(dim=1)

    def plot_uv_diagnostics(self, u_pred, u_true, v_pred, v_true):
        import matplotlib.pyplot as plt
        import os

        # 转为 NumPy
        u_pred_np = u_pred.detach().cpu().numpy().flatten()
        u_true_np = u_true.detach().cpu().numpy().flatten()
        v_pred_np = v_pred.detach().cpu().numpy().flatten()
        v_true_np = v_true.detach().cpu().numpy().flatten()

        u_res = u_pred_np - u_true_np
        v_res = v_pred_np - v_true_np

        # 创建 2×2 图像布局
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # 左上：u 散点图
        axs[0, 0].scatter(u_true_np, u_pred_np, alpha=0.3)
        axs[0, 0].plot(u_true_np, u_true_np, 'r--')
        axs[0, 0].set_title("u: Predicted vs True")
        axs[0, 0].set_xlabel("True u")
        axs[0, 0].set_ylabel("Predicted u")

        # 右上：v 散点图
        axs[0, 1].scatter(v_true_np, v_pred_np, alpha=0.3)
        axs[0, 1].plot(v_true_np, v_true_np, 'r--')
        axs[0, 1].set_title("v: Predicted vs True")
        axs[0, 1].set_xlabel("True v")
        axs[0, 1].set_ylabel("Predicted v")

        # 左下：u 残差直方图
        axs[1, 0].hist(u_res, bins=50, color='blue', alpha=0.7)
        axs[1, 0].set_title("u: Residual Histogram")
        axs[1, 0].set_xlabel("u_pred - u_true")

        # 右下：v 残差直方图
        axs[1, 1].hist(v_res, bins=50, color='green', alpha=0.7)
        axs[1, 1].set_title("v: Residual Histogram")
        axs[1, 1].set_xlabel("v_pred - v_true")

        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "uv_diagnostics.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"📊 综合诊断图已保存到: {fig_path}")

    def plot_residual_histogram(self, u_pred, u_true, v_pred, v_true):
        import matplotlib.pyplot as plt
        import os

        u_res = (u_pred - u_true).detach().cpu().numpy().flatten()
        v_res = (v_pred - v_true).detach().cpu().numpy().flatten()

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].hist(u_res, bins=50, color='blue', alpha=0.7)
        axs[0].set_title("u: Residual Histogram")
        axs[0].set_xlabel("u_pred - u_true")

        axs[1].hist(v_res, bins=50, color='green', alpha=0.7)
        axs[1].set_title("v: Residual Histogram")
        axs[1].set_xlabel("v_pred - v_true")

        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "uv_residual_histogram.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"📉 残差直方图已保存到: {fig_path}")

    def save_velocity_field(self, x, y, u_pred, v_pred, Z0=None, t=None):
        import matplotlib.pyplot as plt
        import pandas as pd
        import os

        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        u_np = u_pred.detach().cpu().numpy().flatten()
        v_np = v_pred.detach().cpu().numpy().flatten()

        # ✅ 保存速度场图（quiver）
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.quiver(x_np, y_np, u_np, v_np, scale=50, width=0.002, color='blue')
        ax.set_title("Velocity Field (u, v)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis('equal')
        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, "velocity_field.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"🖼️ 速度场图已保存到: {fig_path}")

        # ✅ 保存 CSV
        data = {
            'x': x_np,
            'y': y_np,
            'u_pred': u_np,
            'v_pred': v_np
        }
        if Z0 is not None:
            data['Z0'] = Z0.detach().cpu().numpy().flatten()
        if t is not None:
            data['t'] = t.detach().cpu().numpy().flatten()

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.output_dir, "velocity_field.csv")
        df.to_csv(csv_path, index=False)
        print(f"📄 速度场数据已保存到: {csv_path}")

    def save_all(self, original_df, u_pred, v_pred, u_true, v_true,
                 Z0, x, y, z, t, g1, g2,
                 final_losses, extra_metrics=None):
        # 保存 Z₀
        self.save_Z0(x, y, z=z, t=t, u_pred=u_pred, v_pred=v_pred, Z0=Z0)

        # 保存完整预测 CSV
        self.save_full_prediction(original_df, u_pred, v_pred, u_true, v_true, Z0, x, y, z, t)

        # ✅ 保存速度场图和 CSV
        self.save_velocity_field(x, y, u_pred, v_pred, Z0=Z0, t=t)

        # ✅ 统一计算方向一致性指标
        dot_gz, dot_uv = self.compute_direction_metrics(u_pred, v_pred, u_true, v_true, g1, g2, Z0, x, y)

        # 保存方向一致性报告
        self.save_direction_consistency(u_pred, v_pred, u_true, v_true, dot_gz, dot_uv)

        # 保存诊断指标
        self.save_diagnostics(final_losses, g_dot_gradZ0=dot_gz, cos_sim=dot_uv, extra_metrics=extra_metrics)

        #输出综合诊断图像（预测 vs 真实 + 残差直方图）
        self.plot_uv_diagnostics(u_pred, u_true, v_pred, v_true)

        print("所有输出已保存完毕")

