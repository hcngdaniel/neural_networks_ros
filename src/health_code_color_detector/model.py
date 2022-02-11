#!/usr/bin/env python
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 10, (5, 5), stride=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(10, 10, (5, 5), stride=(5, 5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4750, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)

