import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from HillClimbingModel import HillClimbingModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VanillaHillClimbing():

