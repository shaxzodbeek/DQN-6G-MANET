import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingDQN(nn.Module):
    def __init__(self, state_dim=4):
        super(RoutingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
