import torch
import torch.nn as nn
import numpy as np

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first_layer=False):
        super().__init__()
        self.omega_0 = omega_0  # 固定常数，不可学习
        self.linear = nn.Linear(in_features, out_features)
        self.is_first_layer = is_first_layer
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first_layer:
                # 第一层：均匀分布 [-1/in, 1/in]
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                # 非第一层：limit = sqrt(6 / in_features) / omega_0
                limit = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)
            # 偏置初始化为 0
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))