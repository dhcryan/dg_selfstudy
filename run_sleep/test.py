import torch
import torch.nn as nn
from torchsummary import summary
# model = torch.load('C:\\Users\\dhc40\\manyDG\\pre-trained\\sleep40-sleep_dev_100_1682444097.4598415.pt')
# print(model)
# model.cuda()
# print(summary(model, input_size=(256,2,3000)))
import os
import argparse
import utils_sleep
import numpy as np
import time
from model_sleep import SleepBase, SleepDev, SleepCondAdv, SleepDANN, SleepIRM, SleepSagNet, SleepPCL, SleepMLDG
import math
from transformers import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="choose from base, dev, condadv, DANN, IRM, SagNet, PCL, MLDG")
    parser.add_argument('--cuda', type=int, default=0, help="which cuda")
    parser.add_argument('--dataset', type=str, default="sleep", help="dataset name")
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print ('device:', device)
    model_dict = torch.load('C:\\Users\\dhc40\\manyDG\\pre-trained\\sleep40-sleep_dev_100_1682444097.4598415.pt')
    
    if args.model == "base":
        model = SleepBase(device, args.dataset).to(device)
    elif args.model == "dev":
        model = SleepDev(device, args.dataset).to(device)
    
    model.load_state_dict(model_dict)
    # summary(model, (6,2,3,3))
    # print(model)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
#     feature_extractor = Wav2Vec2FeatureExtractor(
#           sampling_rate=24000,
#           truncation=True
# )
    feature_extractor.sampling_rate=100
    print(feature_extractor)