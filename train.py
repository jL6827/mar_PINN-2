#做了多次修正：10.25修正的是：在epoch=7k时，Ro会快速增长；解决吧办法：对Ro用更小的学习率；
#动态调整Ro的权重
#这部分不好将它分成训练和预测的原因是：数据集分割是随机的，所以单独分开比较麻烦

import argparse
import os
import torch
from losses import direction_loss, value_loss, activation_loss
import pandas as pd
import torch.nn.functional as F
from config import ModelConfig
from data_loader import split_dataset_random, load_csv_data_from_df
from physics_model import EnhancedPhysicsInformedThermocline
from compute_approximate_velocity import compute_velocity
from compute_physics_residual import compute_residuals
from physics_residual import geometric_constraint
from utils import get_device, prepare_inputs, LossManager
from save_utils import PredictionSaver

import argparse

def parse_args_for_train():
    p = argparse.ArgumentParser()
    # 默认使用项目内 data/ 和 outputs/ 子目录（相对路径）
    p.add_argument("--input", "-i", default="data/processed_data_mean_train.csv",
                   help="输入 CSV 文件路径（相对项目根目录），默认: data/processed_data_mean_train.csv")
    p.add_argument("--output", "-o", default="outputs/run1",
                   help="训练输出目录（相对项目根目录），默认: outputs/run1")
    p.add_argument("--device", default=None, help="可选：cuda:0 或 cpu")
    return p.parse_args()


def train_prediction_model(input_path, output_dir, device=None):  # 修复：将 - 改为 _
    os.makedirs(output_dir, exist_ok=True)

    # 设备处理：如果用户通过 CLI 传入 device，则使用该设备字符串，否则使用项目自带的 get_device()
    if device is not None:
        device = torch.device(device)
    else:
        device = get_device()

    train_df, test_df = split_dataset_random(input_path)
    t, x, y, z, u_true, v_true, scaler_mgr, original_df = load_csv_data_from_df(train_df, device)

    config = ModelConfig(
        Ro=0.024,
        omega_0=30.0,
        use_scaler=True,
        scaler_mgr=scaler_mgr,
        depth_scaler=scaler_mgr.depth_scaler,
        velocity_scale=1.0,
        grad_clip=10.0
    )
    model = EnhancedPhysicsInformedThermocline(config).to(device)

    # ==================== 初始化 ====================
    base_lr = 1e-4
    ro_base_lr = 5e-4
    #initial_ro_weight = 0.5

    # 参数分离
    ro_params = [p for n, p in model.named_parameters() if n == 'Ro']
    other_params = [p for n, p in model.named_parameters() if n != 'Ro']

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': base_lr},
        {'params': ro_params, 'lr': ro_base_lr}
    ])

    #check_trainable_parameters(model)

    # 损失权重管理
    loss_manager = LossManager({
        'data': 3.5, 'dir': 5, 'phys': 3.0,
        'cont': 1.0, 'geo': 0.2, 'act': 5
    })

    # ==================== 核心辅助函数 ====================
    def get_dynamic_weights(epoch):
        """动态权重调度（分段策略）"""
        if epoch < 4000:
            ro_w = 0.5 * (1 - 0.5 * epoch / 4000)
        # 阶段2：4000-7000个epoch，从0.25衰减到0.1
        elif epoch < 7000:
            ro_w = 0.25 * (1 - 0.6 * (epoch - 4000) / 3000)
        else:
            ro_w = 0.1

        return ro_w, 0.02

    def update_ro_lr(epoch):
        """更新Ro学习率"""
        initial_ro_weight = 0.5  # 第一阶段的初始值
        ro_weight, _ = get_dynamic_weights(epoch)
        lr_scale = max(0.1, ro_weight / initial_ro_weight)

        for pg in optimizer.param_groups:
            if pg['params'][0] is model.Ro:
                pg['lr'] = ro_base_lr * lr_scale
                break

        return lr_scale

    def compute_ro_regularization(model, ro_weight, b_weight):
        """Ro及B标量正则项"""
        ro_min, ro_max = 0.02, 0.1
        ro_reg = ro_weight * (
                F.softplus(50 * (ro_min - model.Ro)) +
                F.softplus(50 * (model.Ro - ro_max))
        )

        b0_reg = b_weight * F.softplus(20 * (model.B0_scalar - 0.05))
        b1_reg = b_weight * F.softplus(20 * (model.B1_scalar - 0.05))

        total_reg = ro_reg + b0_reg + b1_reg
        return total_reg, ro_reg, b0_reg, b1_reg

    # ==================== 训练循环 ====================
    ro_history = []
    num_epochs = 8000

    for epoch in range(num_epochs):
        # 动态学习率和权重
        ro_lr_scale = update_ro_lr(epoch)  # ✅ 获取缩放因子
        ro_weight, b_weight = get_dynamic_weights(epoch)

        # 动态调整物理权重（仅在需要时）
        if epoch >= 3000 and epoch % 500 == 0:
            step = (epoch - 3000) // 500
            loss_manager.set_weights({
                'phys': min(5.0, 3.0 + 0.2 * step),
                'cont': min(2.0, 1.0 + 0.1 * step),
                'geo': min(0.5, 0.2 + 0.05 * step),
                'dir': min(8.0, 5.0 + 0.2 * step)
            })

        optimizer.zero_grad()
        x, y, z, t = prepare_inputs(x, y, z, t)

        # 前向传播
        (u_pred, v_pred, u_bar, v_bar, theta, eta, Z0,
         P, g1, g2, h1, h2, time_phase_C2, time_phase_C3) = compute_velocity(model, x, y, z, t)

        # 计算所有损失
        losses = {
            'data': value_loss(u_pred, v_pred, u_true, v_true),
            'dir': direction_loss(u_pred, v_pred, u_true, v_true),
            'act': activation_loss(theta, eta),
            'geo': geometric_constraint(g1, g2, Z0, x, y),
        }

        # 物理约束损失
        res_u, res_v, res_cont = compute_residuals(model, x, y, z, t)
        losses['phys'] = torch.mean(res_u ** 2 + res_v ** 2)
        losses['cont'] = torch.mean(res_cont ** 2)

        # Z0正则化
        grad_z0 = [torch.autograd.grad(Z0, v, grad_outputs=torch.ones_like(Z0),
                                       create_graph=True)[0] for v in [x, y]]
        z0_reg = 0.5 * torch.mean(Z0) ** 2 + 0.2 * (torch.var(Z0) - 0.1) ** 2 + \
                 0.1 * torch.mean(sum(g ** 2 for g in grad_z0))

        # ✅ 计算Ro正则化（现在返回元组）
        reg_total, ro_reg, b0_reg, b1_reg = compute_ro_regularization(model, ro_weight, b_weight)

        # 总损失
        total_loss = loss_manager.compute_total_loss(losses, epoch=epoch)
        total_loss += reg_total + z0_reg

        # Ro变化率约束（后期稳定）
        ro_change_penalty = torch.tensor(0.0, device=device)  # ✅ 初始化
        if epoch > 7000 and ro_history:
            ro_change_penalty = 0.1 * (model.Ro - ro_history[-1]) ** 2
            total_loss += ro_change_penalty

        total_loss.backward()

        # ⭐ 简化梯度控制：只保留关键部分
        if model.Ro.grad is not None and torch.abs(model.Ro.grad) > 1.0:
            model.Ro.grad.data.clamp_(-0.5, 0.5)

        optimizer.step()
        with torch.no_grad():
            model.Ro.clamp_(1e-2, 0.5)

        if torch.isnan(total_loss):
            print(f"[Epoch {epoch}] NaN detected, stopping.")
            break

        ro_history.append(model.Ro.item())

        # 🎯 完整监控（所有变量都被正确定义）
        if epoch % 1000 == 0:
            # 获取当前学习率
            current_lrs = [f"Ro: {pg['lr']:.2e}" if pg['params'][0] is model.Ro
                           else f"Other: {pg['lr']:.2e}" for pg in optimizer.param_groups]

            print(f"{'=' * 70}")
            print(f"[Epoch {epoch}] 状态监控")
            print(f"{'=' * 70}")

            # 学习率信息
            print(f"  学习率: {', '.join(current_lrs)}")
            print(f"  Ro正则权重: {ro_weight:.4f}, Ro学习率缩放: {ro_lr_scale:.4f}")

            # 网络参数
            print(f"  θ mean: {theta.mean().item():.6f}, η mean: {eta.mean().item():.6f}")
            print(f"  u_pred mean: {u_pred.mean().item():.6f}, v_pred mean: {v_pred.mean().item():.6f}")
            print(f"  Z₀ mean: {Z0.mean().item():.6f}, Z₀ var: {Z0.var().item():.6f}")

            # Ro参数和正则化
            print(f"  Ro value: {model.Ro.item():.6f}")
            print(f"  Ro regularization: {ro_reg.item():.6f}")

            # B标量参数和正则化
            print(f"  B0_scalar: {model.B0_scalar.item():.6f}, B0_reg: {b0_reg.item():.6f}")
            print(f"  B1_scalar: {model.B1_scalar.item():.6f}, B1_reg: {b1_reg.item():.6f}")

            # Ro变化惩罚项（后期）
            if epoch > 7000:
                print(f"  Ro change penalty: {ro_change_penalty.item():.6f}")

            # 各项损失
            print(f"损失分量: ")
            print(f"    Data: {losses['data'].item():.6f}")
            print(f"    Dir:  {losses['dir'].item():.6f}")
            print(f"    Phys: {losses['phys'].item():.6f}")
            print(f"    Cont: {losses['cont'].item():.6f}")
            print(f"    Geo:  {losses['geo'].item():.6f}")
            print(f"    Act:  {losses['act'].item():.6f}")
            print(f"    Z0_reg: {z0_reg.item():.6f}")
            print(f"    Ro_reg: {ro_reg.item():.6f}")
            print(f"  Total Loss: {total_loss.item():.6f}")

            # 梯度信息
            if model.Ro.grad is not None:
                print(f" Ro grad: {model.Ro.grad.item(): .6f} ")

            print(f"{'=' * 70} ")

            # === 保存训练结果 ===
    pd.DataFrame({'Ro': ro_history}).to_csv(os.path.join(output_dir, 'ro_history.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_final.pt'))
    print(f"✅ 模型训练完成，Ro 演化与参数已保存到: {output_dir}")

    summary = [
                  "✅ 模型训练参数记录",
                  f"Ro 初始值: {config.Ro}",
                  f"Ro 最终值: {model.Ro.item():.6f}",
                  f"omega_0: {config.omega_0}",
                  f"velocity_scale: {config.velocity_scale}",
                  f"grad_clip: {config.grad_clip}",
                  f"训练轮数: {num_epochs}",
                  f"基础学习率: 1e-4",
                  f"Ro基准学习率: 5e-4",
                  "损失权重:"
              ] + [f"  {k}: {v}" for k, v in loss_manager.weights.items()] + [
                  "学习率策略: Ro学习率随正则权重同步衰减",
                  "物理权重上限: 5.0 ",
                  "Ro正则约束系数: 50 "
              ]
    theta_abs = torch.mean(torch.abs(theta)).item()
    eta_abs = torch.mean(torch.abs(eta)).item()
    summary += [
        f"theta_abs: {theta_abs:.6f}",
        f"eta_abs: {eta_abs:.6f}"
    ]

    with open(os.path.join(output_dir, "training_summary.txt"), "w") as f:
         f.write("\n".join(summary))

    # === 使用测试集做预测并保存结果 ===
    print("✅ 开始使用测试集进行预测...")

    # 加载测试集数据
    t_test, x_test, y_test, z_test, u_true_test, v_true_test, _, original_test_df = load_csv_data_from_df(test_df,
                                                                                                          device)
    x_test.requires_grad_(True)
    y_test.requires_grad_(True)

    # 执行预测
    (u_pred_test, v_pred_test, u_bar, v_bar, theta, eta, Z0,
     P, g1, g2, h1, h2, time_phase_C2, time_phase_C3) = compute_velocity(model, x_test, y_test, z_test, t_test)

    # 计算损失项
    final_losses = {
        'data': value_loss(u_pred_test, v_pred_test, u_true_test, v_true_test),
        'dir': direction_loss(u_pred_test, v_pred_test, u_true_test, v_true_test),
        'phys': torch.mean(compute_residuals(model, x_test, y_test, z_test, t_test)[0] ** 2 +
                           compute_residuals(model, x_test, y_test, z_test, t_test)[1] ** 2),
        'cont': torch.mean(compute_residuals(model, x_test, y_test, z_test, t_test)[2] ** 2),
        'geo': geometric_constraint(g1, g2, Z0, x_test, y_test),
        'act': activation_loss(theta, eta)
    }

    # ✅ 在这里添加：计算正确的方向角 MAE（带掩码，使用弧度）
    true_mag = torch.sqrt(u_true_test ** 2 + v_true_test ** 2)
    mask = (true_mag > 0.005)  # 与训练时相同的掩码

    pred_angle = torch.atan2(v_pred_test, u_pred_test)
    true_angle = torch.atan2(v_true_test, u_true_test)  # ✅ 修正：u_true_test

    # 计算角度差（弧度）
    angle_diff = torch.abs(pred_angle - true_angle)

    # 处理角度差超过 π 的情况（方向相反）
    angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)

    # 仅在速度大的区域计算 MAE（弧度）
    direction_mae_rad = angle_diff[mask].mean().item()

    # 将正确的 MAE 添加到 final_losses 中（用于保存）
    final_losses['direction_mae'] = torch.tensor(direction_mae_rad)

    saver = PredictionSaver(model, scaler_mgr, output_dir)
    saver.save_all(
        original_df=original_test_df,
        u_pred=u_pred_test, v_pred=v_pred_test,
        u_true=u_true_test, v_true=v_true_test,
        Z0=Z0, x=x_test, y=y_test, z=z_test, t=t_test,
        g1=g1, g2=g2,
        final_losses=final_losses,
        extra_metrics={'Ro': model.Ro.item()}
    )

    print("✅ 测试集预测完成，结果已保存到:", output_dir)

    print("🔄 Start drawing comparison plots...")

    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy arrays for plotting
    u_pred_np = u_pred_test.detach().cpu().numpy().flatten()
    v_pred_np = v_pred_test.detach().cpu().numpy().flatten()
    u_true_np = u_true_test.detach().cpu().numpy().flatten()
    v_true_np = v_true_test.detach().cpu().numpy().flatten()

    indices = np.arange(len(u_pred_np))  # 使用全部测试点

    # Create three separate figures
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))

    # Plot 1: u_pred vs u_true
    axes[0].plot(indices, u_true_np[indices], 'b-', linewidth=1.5, alpha=0.8, label='u_true')
    axes[0].plot(indices, u_pred_np[indices], 'r-', linewidth=1.5, alpha=0.8, label='u_pred')
    axes[0].set_xlabel('Data Point Index')
    axes[0].set_ylabel('u Velocity')
    axes[0].set_title(f'u_pred vs u_true Comparison (All test Points)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: v_pred vs v_true
    axes[1].plot(indices, v_true_np[indices], 'g-', linewidth=1.5, alpha=0.8, label='v_true')
    axes[1].plot(indices, v_pred_np[indices], 'm-', linewidth=1.5, alpha=0.8, label='v_pred')
    axes[1].set_xlabel('Data Point Index')
    axes[1].set_ylabel('v Velocity')
    axes[1].set_title(f'v_pred vs v_true Comparison (All test Points)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Calculate true direction angles
    true_magnitude = np.sqrt(u_true_np ** 2 + v_true_np ** 2)
    true_u_norm = u_true_np / (true_magnitude + 1e-8)  # Avoid division by zero
    true_v_norm = v_true_np / (true_magnitude + 1e-8)
    true_angle = np.arctan2(true_v_norm, true_u_norm) * 180 / np.pi  # Convert to degrees

    # Calculate predicted direction angles
    pred_magnitude = np.sqrt(u_pred_np ** 2 + v_pred_np ** 2)
    pred_u_norm = u_pred_np / (pred_magnitude + 1e-8)
    pred_v_norm = v_pred_np / (pred_magnitude + 1e-8)
    pred_angle = np.arctan2(pred_v_norm, pred_u_norm) * 180 / np.pi

    axes[2].plot(indices, true_angle[indices], 'c-', linewidth=1.5, alpha=0.8, label='True Direction Angle')
    axes[2].plot(indices, pred_angle[indices], 'y-', linewidth=1.5, alpha=0.8, label='Predicted Direction Angle')
    axes[2].set_xlabel('Data Point Index')
    axes[2].set_ylabel('Direction Angle (degrees)')
    axes[2].set_title(f'Velocity Direction Angle Comparison (All test Points)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    comparison_plot_path = os.path.join(output_dir, 'velocity_comparison_plots.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save comparison data for the selected points
    # --- 原有代码：到保存 comparison_data 为止，保持不变 ---

    comparison_data = pd.DataFrame({
        'index': indices,
        'u_true': u_true_np[indices],
        'u_pred': u_pred_np[indices],
        'v_true': v_true_np[indices],
        'v_pred': v_pred_np[indices],
        'true_angle_deg': true_angle[indices],
        'pred_angle_deg': pred_angle[indices],
        'angle_difference_deg': np.abs(true_angle[indices] - pred_angle[indices])
    })

    comparison_csv_path = os.path.join(output_dir, 'velocity_comparison_data.csv')
    comparison_data.to_csv(comparison_csv_path, index=False)

    # 添加掩码，只在速度大的区域计算; 提取用于绘图的子集数据
    u_true_plot = u_true_np[indices]
    v_true_plot = v_true_np[indices]
    u_pred_plot = u_pred_np[indices]
    v_pred_plot = v_pred_np[indices]

    # 计算真实速度大小
    true_mag_plot = np.sqrt(u_true_plot ** 2 + v_true_plot ** 2)
    mask_plot = (true_mag_plot > 0.005)  # 与训练时完全一致的掩码

    # 计算 MAE（仅在掩码区域内）
    u_mae = np.mean(np.abs(u_pred_plot[mask_plot] - u_true_plot[mask_plot]))
    v_mae = np.mean(np.abs(v_pred_plot[mask_plot] - v_true_plot[mask_plot]))

    # 处理角度差（避免 350° 和 10° 的差为 340°）
    angle_diff = np.abs(true_angle[indices] - pred_angle[indices])
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)  # 正确处理角度环绕
    angle_mae = np.mean(angle_diff[mask_plot])

    # --- 原有绘图代码不变，仅统计输出使用 mask ---
    print(f"\n📊 Prediction Accuracy Statistics (for all {len(u_pred_np)} test points):")
    print(f"  u-component MAE: {u_mae:.6f}")
    print(f"  v-component MAE: {v_mae:.6f}")
    print(f"  Direction Angle MAE: {angle_mae:.2f}°")
    print(f"  Comparison data saved to: {comparison_csv_path}")
    print(f"  Comparison plots saved to: {comparison_plot_path}")

if __name__ == "__main__":
    args = parse_args_for_train()
    # 若希望确保 output 路径始终在仓库脚本目录下（即不受当前工作目录影响），可以把下面一行替换为：
    # repo_root = os.path.dirname(os.path.abspath(__file__))
    # output_dir = os.path.join(repo_root, args.output)
    output_dir = args.output
    input_path = args.input
    train_prediction_model(input_path=input_path, output_dir=output_dir, device=args.device)