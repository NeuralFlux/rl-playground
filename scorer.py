"""
Input Size: 13 x [Suit (One-hot) [4] + Rank (One-hot) [12]]
Output Size: 13 x [Set (One-hot) [4] + Index of Card from Input (One-hot) [12]]

Suit = (0, 1, 2, 3, Joker) [5] - [1] = 4
Set = (0, 1, 2, 3, Nil) [5] - [1] = 4
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ScoringNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ScoringNet, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))


class Scorer(object):
    def __init__(self) -> None:
        pass

