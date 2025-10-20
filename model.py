# model.py
# Mendefinisikan arsitektur neural network untuk policy.

import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Jaringan syaraf tiruan yang memetakan state ke sebuah skor (logit)
    untuk setiap kemungkinan aksi (kandidat evakuasi).
    """

    def __init__(self, input_dim=6, hidden_dim=128, output_dim=1):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass melalui jaringan."""
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.out(x)
