import torch
from torchsummary import summary
import sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd
from collections import OrderedDict
sys.path.append('C:\\Users\\dhc40\\manyDG')
sys.path.append('C:\\Users\\dhc40\\manyDG\\data\\sleep')
sys.path.append('C:\\Users\\dhc40\\manyDG\\data\\HealthDG')
from model import Base, Dev, CondAdv, DANN, IRM, SagNet, PCL, MLDG

model=Base()
model_state_dict = torch.load("C:\\Users\\dhc40\\manyDG\\pre-trained\\sleep0-sleep_dev_100_1679541603.0450962.pt")
# model=model.load_state_dict(model)
model.load_state_dict(model_state_dict)

def summarize_model(model):
    summary(model, device="cuda")

model.cuda()
summarize_model(model)
