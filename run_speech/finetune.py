import os
import argparse
import utils_sleep
import torch
import numpy as np
import time
import sys

from model_sleep import SleepBase, SleepDev, SleepCondAdv, SleepDANN, SleepIRM, SleepSagNet, SleepPCL, SleepMLDG

import torch
import torchaudio

sys.path.append('C:\\Users\\dhc40\\manyDG')
sys.path.append('C:\\Users\\dhc40\\manyDG\\data\\sleep')


if __name__ == '__main__':
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def trainloader_for_dev():
        train_X = []
        train_X_aux = []
        for i, (_, X) in enumerate(train_pat_map.items()):
            np.random.shuffle(X)
            train_X += X[:len(X)//2 + 1]
            train_X_aux += X[-len(X)//2 - 1:]
        train_loader = torch.utils.data.DataLoader(utils_sleep.SleepDoubleLoader(train_X, train_X_aux),
                batch_size=256, shuffle=True, num_workers=16)
        return train_loader
    # print(device)
    train_loader_dev=trainloader_for_dev()
    bundle = torchaudio.pipelines.WAV2VEC2_BASE