import torch
import torch.nn as nn

class DynamicZ0Net(nn.Module):
    def __init__(self, base_depth=73.0, depth_range=15.0):
        super().__init__()
        self.base_depth = base_depth
        self.depth_range = depth_range

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

        self.learnable_base = nn.Parameter(torch.tensor(base_depth))
        self.learnable_range = nn.Parameter(torch.tensor(depth_range))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.3)

    def forward(self, t_norm, x_norm, y_norm):
        inputs = torch.cat([t_norm, x_norm, y_norm], dim=1)
        z0_variation = self.net(inputs)
        final_z0 = self.learnable_base + (z0_variation * self.learnable_range / 2)
        final_z0 = torch.clamp(final_z0, 65.0, 85.0)
        return final_z0
