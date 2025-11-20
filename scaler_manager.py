import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ScalerManager:
    def __init__(self):
        self.full_scaler = MinMaxScaler()
        self.depth_scaler = MinMaxScaler()

    def fit(self, data_frame):
        # 假设列顺序为 ['t', 'x', 'y', 'z']
        self.full_scaler.fit(data_frame[['time', 'longitude', 'latitude', 'depth']])
        self.depth_scaler.fit(data_frame[['depth']])

    def transform_all(self, data_frame):
        return self.full_scaler.transform(data_frame[['time', 'longitude', 'latitude', 'depth']])

    def inverse_transform_all(self, norm_array):
        return self.full_scaler.inverse_transform(norm_array)

    def transform_depth(self, z_array):
        return self.depth_scaler.transform(z_array.reshape(-1, 1))

    def inverse_transform_depth(self, z_norm_array):
        z_np = z_norm_array.detach().cpu().numpy().reshape(-1, 1)
        return self.depth_scaler.inverse_transform(z_np)

    def inverse_x(self, x_norm_tensor):
        x_np = x_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        # 构造一个虚拟输入，填充其他列
        dummy = np.zeros((x_np.shape[0], 4))
        dummy[:, 1] = x_np[:, 0]  # x 是第2列（索引1）
        return self.full_scaler.inverse_transform(dummy)[:, 1]  # 返回反变换后的 x

    def inverse_y(self, y_norm_tensor):
        y_np = y_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        dummy = np.zeros((y_np.shape[0], 4))
        dummy[:, 2] = y_np[:, 0]  # y 是第3列（索引2）
        return self.full_scaler.inverse_transform(dummy)[:, 2]  # 返回反变换后的 y

    def inverse_time(self, t_norm_tensor):
        t_np = t_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        dummy = np.zeros((t_np.shape[0], 4))
        dummy[:, 0] = t_np[:, 0]  # time 是第1列（索引0）
        return self.full_scaler.inverse_transform(dummy)[:, 0]

    def inverse_velocity(self, vel_norm_tensor):
        vel_np = vel_norm_tensor.detach().cpu().numpy().reshape(-1, 1)
        # 如果你对速度只做了缩放，可以直接反变换；否则需要额外 scaler
        return vel_np  # 或使用 self.velocity_scaler.inverse_transform(vel_np)
