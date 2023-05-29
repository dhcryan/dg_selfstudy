import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch import autograd
import math
from torch.nn import Parameter
import matplotlib.pyplot as plt
import torchaudio
from transformers import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from torch.nn import Parameter
"""
Residual block
"""
class ResBlock_sleep(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock_sleep, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)     
        self.downsample = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
           nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


"""
SoftCEL Loss
"""
def SoftCEL2(y_hat, y):
    # weighted average version
    p = F.log_softmax(y_hat, 1)
    return - torch.mean((p*y).sum(1) / y.sum(1))

def SoftCEL(y_hat, y):
    p = F.log_softmax(y_hat, 1)
    y2 = torch.where(y >= torch.max(y, 1, keepdim=True)[0].repeat(1, 6) * 0.5, torch.ones_like(y), torch.zeros_like(y))
    return - torch.mean((p*y2).sum(1) / y2.sum(1))


"""
FeatureCNN
"""

class FeatureCNN_sleep(nn.Module):
    def __init__(self, n_dim=128):
        super(FeatureCNN_sleep, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1), # change the argument for dimension of input channels
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock_sleep(6, 8, 2, True, False)
        self.conv3 = ResBlock_sleep(8, 16, 2, True, True)
        self.conv4 = ResBlock_sleep(16, 32, 2, True, True)
        self.n_dim = n_dim
        
    def torch_stft(self, X_train):
        signal = []
        for s in range(X_train.shape[1]):
            spectral = torch.stft(X_train[:, s, :],
                n_fft = 256,
                hop_length = 256 * 1 // 4,
                center = False,
                onesided = True,
                return_complex=False)
            signal.append(spectral)
        
        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)
        signal = (signal1 ** 2 + signal2 ** 2) ** 0.5
        return torch.clip(torch.log(torch.clip(signal, min=1e-8)), min=0)

    @staticmethod
    def functional_res_block(x, conv1_weight, conv1_bias, bn1_weight, bn1_bias, \
            conv2_weight, conv2_bias, bn2_weight, bn2_bias, ds_conv_weight, ds_conv_bias,
            ds_bn_weight, ds_bn_bias, pooling=True):
        out = F.conv2d(x, conv1_weight, conv1_bias, stride=2, padding=1)
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=bn1_weight, bias=bn1_bias, training=True)
        out = F.elu(out)
        out = F.conv2d(out, conv2_weight, conv2_bias, padding=1)
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=bn2_weight, bias=bn2_bias, training=True)
        residual = F.conv2d(x, ds_conv_weight, ds_conv_bias, stride=2, padding=1)
        residual = F.batch_norm(residual, running_mean=None, running_var=None, weight=ds_bn_weight, bias=ds_bn_bias, training=True)
        out += residual
        if pooling:
            out = F.max_pool2d(out, 2, stride=2)
        out = F.dropout(out, 0.5)
        return out
    
    def functional_forward(self, x, fast_weights):
        x = self.torch_stft(x)
        fast_weights_ls = list(fast_weights.values())
        out = F.conv2d(x, fast_weights_ls[0], fast_weights_ls[1], stride=1, padding=1)
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=fast_weights_ls[2], bias=fast_weights_ls[3], training=True)
        out = F.elu(out)
        out = self.functional_res_block(out, *fast_weights_ls[4:16], False)
        out = self.functional_res_block(out, *fast_weights_ls[16:28])
        out = self.functional_res_block(out, *fast_weights_ls[28:40])
        out = out.reshape(out.shape[0], -1)
        return out

    def forward(self, x):
        x = self.torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)
        return x 
"""
Core Module
"""

"""
Wav2vec 2.0
"""
# WAV2VEC2를 freezing하여서 feature extractor로 사용
class PretrainedWav2Vec2Model(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()

        self.sample_rate = sample_rate

        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = self.bundle.get_model()
        # model.eval()
        self.model = model
        self.model.feature_extractor.requires_grad_(False)  # feature extractor freezing

    def forward(self, x):
        # print(x.shape)==torch.Size([256, 2, 3000])
        # x[:,0] is the first channel-> [256, 3000]
        x = torchaudio.functional.resample(x[:,0], self.sample_rate, self.bundle.sample_rate)
        # print(x.shape) => 256,900
        # for param in self.model.feature_extractor.parameters():
        #     param.requires_grad = False       
        c, _ = self.model.extract_features(x)
        # print(c.shape)
        # print(c[-1].shape)
        return c[-1]

class Base(nn.Module):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(Base, self).__init__()
        self.dataset = dataset
        self.model = model
        if dataset == "sleep":
            #100 hz==10000 sample rate
            self.wav2vec2 = PretrainedWav2Vec2Model(10000)
            self.feature_cnn = FeatureCNN_sleep()
            self.g_net = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 5), # change the argument for dimension of output
            )
            #wav2vec의 모델의  dimension이 128이 아니기 때문에 맞춰줘야함
            self.fc3 = nn.Linear(768, 1)
            self.fc4 = nn.Linear(14, 128)

    @staticmethod
    def model_initialization_params():
        idx_path = './data/idxFile'
        diagstring2idx = pickle.load(open('{}/diagstring2idx.pkl'.format(idx_path), 'rb'))
        diagICD2idx = pickle.load(open('{}/diagICD2idx.pkl'.format(idx_path), 'rb'))
        labname2idx = pickle.load(open('{}/labname2idx.pkl'.format(idx_path), 'rb'))
        physicalexam2idx = pickle.load(open('{}/physicalexam2idx.pkl'.format(idx_path), 'rb'))
        treatment2idx = pickle.load(open('{}/treatment2idx.pkl'.format(idx_path), 'rb'))
        medname2idx = pickle.load(open('{}/medname2idx.pkl'.format(idx_path), 'rb'))
        return diagICD2idx, diagstring2idx, labname2idx, physicalexam2idx, treatment2idx, medname2idx

    def forward(self, x):
        # x = self.wav2vec2(x)
        # x = self.fc3(x)
        # x = x.squeeze(axis=2)
        # x = self.fc4(x)
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out, x

class BYOL(torch.nn.modules.loss._Loss):
    """
    "boost strap your own latent", search this paper
    """
    def __init__(self, device, T=0.5):
        """
        T: softmax temperature (default: 0.07)
        """
        super(BYOL, self).__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive, y_par=None, threshold=1.0):
        # L2 normalize
        emb_anchor = nn.functional.normalize(emb_anchor, p=2, dim=1)
        emb_positive = nn.functional.normalize(emb_positive, p=2, dim=1)
        # compute the cosine similarity
        if y_par is None:
            # the original BYOL version
            l_pos = torch.einsum('nc,nc->n', [emb_anchor, emb_positive]).unsqueeze(-1)
        else:
            # we select the pairs to compute the loss based on label similarity
            l_pos = torch.einsum('nc,nc,n->n', [emb_anchor, emb_positive, y_par]).unsqueeze(-1)
        loss = - torch.clip(l_pos, max=threshold).mean()
        return loss

def vec_minus(v, z):
    z = F.normalize(z, p=2, dim=1)
    return v - torch.einsum('nc,nc,nd->nd', v, z, z)

class GNet(nn.Module):
    """ prototype based predictor for multi-class classification """
    def __init__(self, N, dim):
        super(GNet, self).__init__()
        self.N = N
        self.dim = dim
        self.prototype = nn.Parameter(torch.randn(N, dim))
        self.prototype.requires_grad = True
        self.T = 0.5
        #self.fc = nn.Linear(dim, N)
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        logits = x @ F.normalize(self.prototype, p=2, dim=1).T / self.T
        #logits = self.fc(x)
        return logits

class GNet_binary(nn.Module):
    """ prototype based predictor for binary classification """
    def __init__(self, N, dim):
        super(GNet, self).__init__()
        self.N = N
        self.prototype = nn.Parameter(torch.randn(N, dim))
        self.prototype.requires_grad = True
        self.T = 0.5
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        logits = x @ F.normalize(self.prototype, p=2, dim=1).T / self.T
        return torch.softmax(logits, 1)[:, 0]

class Dev(Base):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(Dev, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        if dataset == "sleep":
            self.q_net = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.p_net = nn.Sequential(
                nn.Linear(128 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
            self.g_net = GNet(5, 128) # change the argument for dimension of output
            #self.fc1 = nn.Linear(23, 1) # change the argument for dimension of input channels
            #self.fc2 = nn.Linear(2048, 128) # change the argument for sequence length of input
    def forward(self, x):
        """
        feature CNN is h(x)
        proj is q(x)
        predictor is g(x)
        mutual reconstruction p(x)
        """
        #x = x.transpose(1, 2)
        #v = self.fc1(x)
        #v = v.squeeze(axis=2)
        #v = self.fc2(v)
        v = self.wav2vec2(x)
        # print(v.shape)
        v = self.fc3(v)
        # print(v.shape)
        v = v.squeeze(axis=2)
        # print(v.shape)
        v = self.fc4(v)
        # print(v.shape)
        #v = self.feature_cnn(x)
        z = self.q_net(v)
        e = vec_minus(v, z)
        out = self.g_net(e)
        # rec = self.p_net(z)
        return out, z, z, v, e
