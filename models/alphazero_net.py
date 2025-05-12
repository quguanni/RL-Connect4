import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 6 * 7, 7)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 7, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        return policy, value.squeeze(1)