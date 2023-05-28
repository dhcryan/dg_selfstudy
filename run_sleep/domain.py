import utils_sleep   
import os
import argparse
import utils_sleep
import torch
import numpy as np
import time
from model_sleep import SleepBase, SleepDev, SleepCondAdv, SleepDANN, SleepIRM, SleepSagNet, SleepPCL, SleepMLDG
 
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print ('device:', device)
    # set random seed
seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True
path = "C:\\Users\\dhc40\\manyDG\\data\\sleep\\test_pat_map_sleep.pkl"

if os.path.exists(path):
    train_pat_map, test_pat_map, val_pat_map = utils_sleep.load()
else:   
    train_pat_map, test_pat_map, val_pat_map = utils_sleep.load_and_dump()
    
def trainloader_for_dev():
        train_X = []
        train_X_aux = []
        for i, (_, X) in enumerate(train_pat_map.items()):
            # if i == args.N_pat: break
            np.random.shuffle(X)
            train_X += X[:len(X)//2 + 1]
            train_X_aux += X[-len(X)//2 - 1:]
        
        train_loader = torch.utils.data.DataLoader(utils_sleep.SleepDoubleLoader(train_X, train_X_aux),
                batch_size=256, shuffle=True, num_workers=16)
        return train_loader       
train_loader_dev=trainloader_for_dev()
print(train_loader_dev)