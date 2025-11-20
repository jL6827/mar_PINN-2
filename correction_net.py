import torch
import torch.nn as nn

class DepthAwareCorrectionNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, init_bias=1.0):
        super().__init__()
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )

        self.depth_specific_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim)
        )

        self.fusion_weight = nn.Parameter(torch.tensor(0.7))
        self.depth_scaler = None
        self._initialize_weights(init_bias)

    def _initialize_weights(self, init_bias):
        for m in self.main_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, init_bias)
        for m in self.depth_specific_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.5)

    def forward(self, x_norm, y_norm, depth_norm, t_norm, Z0_enhanced):
        base_features = torch.cat([x_norm, y_norm, depth_norm, t_norm], dim=1)
        base_correction = self.main_net(base_features)

        if self.depth_scaler is None:
            depth_physical = depth_norm
        else:
            depth_physical_np = self.depth_scaler.inverse_transform(depth_norm.cpu().detach().numpy())
            depth_physical = torch.FloatTensor(depth_physical_np).to(depth_norm.device)

        relative_depth = depth_physical - Z0_enhanced
        depth_gradient = torch.exp(-torch.abs(relative_depth) / 10.0)
        depth_features = torch.cat([relative_depth, depth_gradient], dim=1)
        depth_specific_correction = self.depth_specific_net(depth_features)

        fusion_weight = torch.sigmoid(self.fusion_weight)
        final_correction = fusion_weight * base_correction + (1 - fusion_weight) * depth_specific_correction
        return final_correction
