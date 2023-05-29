
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd
from collections import OrderedDict
import sys
import os
log_path='C:\\Users\\dhc40\\manyDG\\run_speech\\Result'

from torch.utils.tensorboard import SummaryWriter
writer2_loss = SummaryWriter(log_dir=log_path)

sys.path.append('C:\\Users\\dhc40\\manyDG\\data\\sleep')
sys.path.append('C:\\Users\\dhc40\\manyDG\\run_speech')

from model import Base, Dev
from model import SoftCEL2, BYOL

SoftCEL = nn.CrossEntropyLoss()

"""
EEG sleep staging
"""
class SleepBase(nn.Module):
    def __init__(self, device, dataset):
        super(SleepBase, self).__init__()
        self.model = Base(dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.dataset = dataset

    def train(self, train_loader, device):
        self.model.train()
        loss_collection = []
        count = 0
        for X, y in train_loader:
            convScore, _ = self.model(X.to(device))
            y = y.to(device)
            count += convScore.shape[0]
            loss = nn.CrossEntropyLoss()(convScore, y)
            loss_collection.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print ('train avg loss: {:.4}, count: {}'.format(sum(loss_collection) / len(loss_collection), len(loss_collection)))
            
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                convScore, _ = self.model(X.to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())
        return result, gt

class SleepDev(nn.Module):
    def __init__(self, device, dataset):
        super(SleepDev, self).__init__()
        self.model = Dev(dataset)
        self.BYOL = BYOL(device)
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        
    def train_base(self, train_loader, device):
        self.model.train()
        loss_collection = []
        for X, y in train_loader:
            out, _, _, _, _ = self.model(X.to(device))
            y = y.to(device)
            loss = nn.CrossEntropyLoss()(out, y)
            loss_collection.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print ('train avg loss: {:.4}, count: {}'.format(sum(loss_collection) / len(loss_collection), len(loss_collection)))
    
    def train(self, train_loader, device,i):
        self.model.train()
        loss_collection = [[], [], [], [], []]

        for _, (X, X2, label, label2) in enumerate(train_loader):
            X = X.to(device)
            X2 = X2.to(device)
            label = label.to(device)
            label2 = label2.to(device)
            # domain=domain
            y_par = (label != label2).float()

            out, _, z, v, e = self.model(X)
            out2, _, z2, v2, e2 = self.model(X2)

            ##### build rec
            prototype = self.model.g_net.prototype[label]
            prototype2 = self.model.g_net.prototype[label2]
            rec = self.model.p_net(torch.cat([z, prototype2], dim=1))
            rec2 = self.model.p_net(torch.cat([z2, prototype], dim=1))
            ######

            # loss1: cross entropy loss
            loss1 = nn.CrossEntropyLoss()(out, label) + nn.CrossEntropyLoss()(out2, label2)
            # loss2: same latent factor
            loss2 = self.BYOL(z, z2)
            # loss3: reconstruction loss
            loss3 = self.BYOL(rec, v2) + self.BYOL(rec2, v)
            # loss4: same embedding space
            z_mean = (torch.mean(z, dim=0) + torch.mean(z2, dim=0)) / 2.0
            v_mean = (torch.mean(v, dim=0) + torch.mean(v2, dim=0)) / 2.0
            loss4 = torch.sum(torch.pow(z_mean - v_mean, 2)) / torch.sum(torch.pow(v_mean.detach(), 2))
            # loss5: supervised contrastive loss
            sim = F.normalize(e, p=2, dim=1) @ F.normalize(e2, p=2, dim=1).T
            y_cross = (label.reshape(-1, 1) == label2.reshape(1, -1)).float().to(device)
            avg_pos = - torch.sum(sim * y_cross) / torch.sum(y_cross)
            avg_neg = torch.sum(sim * (1 - y_cross)) / torch.sum(1 - y_cross)
            loss5 = avg_pos + avg_neg
            loss = 1 * loss1  + 1 * loss2 + 1 * loss3 + 1 * loss4 + 0 * loss5
            
            # writer.add_scalar(".\\Loss\\train", loss)
            writer2_loss.add_scalars("Loss", {"total": loss,
                                        "cross entropy": loss1,
                                        "MMD": loss2,
                                        "reconstruction": loss3,
                                        "similarity": loss4}, i)            
            loss_collection[0].append(loss1.item())
            loss_collection[1].append(loss2.item())
            loss_collection[2].append(loss3.item())
            loss_collection[3].append(loss4.item())
            loss_collection[4].append(loss5.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print ('train avg loss: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, count: {}'.format(
                np.sum(loss_collection) / len(train_loader), 
                sum(loss_collection[0]) / len(train_loader), 
                sum(loss_collection[1]) / len(train_loader), 
                sum(loss_collection[2]) / len(train_loader), 
                sum(loss_collection[3]) / len(train_loader), 
                sum(loss_collection[4]) / len(train_loader), 
                len(train_loader)
                )
            )   
      
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for X, y in test_loader:
                out, _, _, _, _ = self.model(X.to(device))
                result = np.append(result, torch.max(out, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())
        return result, gt
    writer2_loss.flush()
    writer2_loss.close()
