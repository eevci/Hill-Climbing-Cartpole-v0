import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from HillClimbingModel import HillClimbingModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VanillaHillClimbingAgent():
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.policy = HillClimbingModel(stateSize, actionSize)
        self.bestReward = -np.Inf
        self.bestWeight = self.policy.getWeights()
        self.noiseScale = 5e-1
        self.gamma = 1.0

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state).detach().cpu().numpy().squeeze()
        action = np.argmax(probs)  #stochastic policy
        return action

    def update(self, rewards):
        discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
        currentReward = sum([a * b for a, b in zip(discounts, rewards)])

        if currentReward >= self.bestReward:  # found better weights
            self.bestReward = currentReward
            self.bestWeight = self.policy.getWeights()

        newWeights = self.bestWeight[0] + self.noiseScale * np.random.rand(*self.bestWeight[0].shape)
        newBiases = self.bestWeight[1] + self.noiseScale * np.random.rand(*self.bestWeight[1].shape)
        self.policy.setWeights([newWeights, newBiases])



