import numpy as np
import torch
import torch.nn as nn
from HillClimbingModel import HillClimbingModel
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AdaptiveNoiseHillClimbingAgent():

    def __init__(self, stateSize, actionSize):
        """Implementation of hill climbing with adaptive noise scaling.
            stateSize = Size of the state
            actionSize = Size of the action
            policy = Simple Hill Climbing model
            bestReward = Reward to compare if current policy is better
            bestWeight = Weights that gives the higher scores
            noise_scale (float): standard deviation of additive noise
            gamma (float): discount rate
        """
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.policy = HillClimbingModel(stateSize, actionSize)
        self.bestReward = -np.Inf
        self.bestWeight = self.policy.getWeights()
        self.noiseScale = 1e-2
        self.minNoiceScale = 1e-3
        self.maxNoiceScale = 2
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
            self.noiseScale = max(self.minNoiceScale, self.noiseScale / 2)
        else:  # did not find better weights
            self.noiseScale = min(self.maxNoiceScale, self.noiseScale * 2)
        newWeights = self.bestWeight[0] + self.noiseScale * np.random.rand(*self.bestWeight[0].shape)
        newBiases = self.bestWeight[1] + self.noiseScale * np.random.rand(*self.bestWeight[1].shape)
        self.policy.setWeights([newWeights, newBiases])
