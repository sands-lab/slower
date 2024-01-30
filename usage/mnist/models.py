import torch.nn as nn

from usage.common.helper import seed


class ServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        seed()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        seed()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x
