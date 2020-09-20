import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HillClimbingModel(nn.Module):
    def __init__(self, stateSize, actionSize):
        super(HillClimbingModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(stateSize, actionSize)),
            ('sig', nn.Sigmoid())
        ])).to(device)

    def forward(self, state):
        return self.model(state)

    def setWeights(self, weights):
        """override current weights with given weight list values"""
        weights = iter(weights)
        for param in self.model.parameters():
            param.data.copy_(torch.from_numpy(next(weights)))

    def getWeights(self):
        """get current weights as list"""
        weights = []
        for param in self.model.parameters():
            weights.append(param.detach().cpu().numpy())
        return weights