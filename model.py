import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch import autograd
import math
from transformers import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
from torch.nn import Parameter
# from transformers import Wav2Vec2Model
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
    # weighted average version, but considers the large votes
    # for example, the vote distribution is [8, 5, 4, 1, 1, 1]
    # instead of using [8, 5, 4, 1, 1, 1] as weights, we consider the votes that not smaller than
    # half of the largest weight and then use [1, 1, 1, 0, 0, 0] as the weights
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
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1),
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
new feature extractor
"""


# class FeatureWav2Vec2_sleep(nn.Module):
#     def __init__(self, wav2vec2_model_name):
#         super(FeatureWav2Vec2_sleep, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        
#     def forward(self, input_ids):
#         features = self.wav2vec2(input_ids).last_hidden_state
#         # reshape features if needed
#         features = features.view(features.size(0), -1)
#         return features


"""
Core Module
"""
class Base(nn.Module):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(Base, self).__init__()
        self.dataset = dataset
        self.model = model
        if dataset == "sleep":
            self.feature_cnn=FeatureCNN_sleep()
            # self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
            # self.feature_extractor.sampling_rate=100
            # self.feature_cnn=self.feature_extractor
            self.g_net = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
            )
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
        self.prototype = nn.Parameter(torch.randn(N, dim))
        self.prototype.requires_grad = True
        self.T = 0.5
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        logits = x @ F.normalize(self.prototype, p=2, dim=1).T / self.T
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
            self.g_net = GNet(5, 128)



    def forward(self, x):
        """
        feature CNN is h(x)
        proj is q(x)
        predictor is g(x)
        mutual reconstruction p(x)
        """
        # print(x.shape)
        v = self.feature_cnn(x)
        # print(v)
        z = self.q_net(v)
        e = vec_minus(v, z)
        out = self.g_net(e)
        # rec = self.p_net(z)
        return out, z, z, v, e

def entropy_for_CondAdv(labels, base=None):
    """ Computes entropy of label distribution. """
    from math import log, e
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent

class CondAdv(Base):
    """ ICML 2017 Rf-radio """
    def __init__(self, dataset, N_pat, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(CondAdv, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        if dataset == "sleep":
            self.discriminator = nn.Sequential(
                nn.Linear(128 + 5, 128),
                nn.ReLU(),
                nn.Linear(128, N_pat)
            )

    def forward(self, x):
        rep = self.feature_cnn(x)
        out = self.g_net(rep)
        rep = torch.cat([rep, out.detach()], dim=1)
        d_out = self.discriminator(rep)
        return out, d_out, rep

    def forward_with_rep(self, rep):
        d_out = self.discriminator(rep)
        return d_out

class ReverseLayerF(autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANN(Base):
    """ ICML 2015, JMLR 2016 """
    def __init__(self, dataset, N_pat, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(DANN, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        if dataset == "sleep":
            self.discriminator = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, N_pat)
            )

    def forward(self, x, alpha):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        d_out = self.discriminator(reverse_feature)
        return out, d_out

class IRM(Base):
    """ arXiv 2020 """
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(IRM, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
    
    def forward(self, x):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out

class SagNet(Base):
    """ CVPR 2021 """
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super(SagNet, self).__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        self.dataset = dataset
        if dataset == "sleep":
            self.d_layer3 = ResBlock_sleep(16, 32, 2, True, True)
            self.d_net = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 5)
            )

    def sleep_pre_layers(self, x):
        x_random = x[torch.randperm(x.size()[0]), :, :]
        x = self.feature_cnn.torch_stft(x)
        x = self.feature_cnn.conv1(x)
        x = self.feature_cnn.conv2(x)
        x = self.feature_cnn.conv3(x)

        x_random = self.feature_cnn.torch_stft(x_random)
        x_random = self.feature_cnn.conv1(x_random)
        x_random = self.feature_cnn.conv2(x_random)
        x_random = self.feature_cnn.conv3(x_random)
        return x, x_random

    def sleep_post_layers(self, SR_rep, CR_rep):
        SR_rep = self.feature_cnn.conv4(SR_rep)
        out = self.g_net(SR_rep.reshape(SR_rep.size(0), -1))

        CR_rep = self.d_layer3(CR_rep)
        out_random = self.d_net(CR_rep.reshape(CR_rep.size(0), -1))
        return out, out_random

    def forward_train(self, x):
        if self.dataset == "sleep":
            x, x_random = self.sleep_pre_layers(x)
        # get statisics
        x_mean = torch.mean(x, keepdim=True, dim=(2, 3))
        x_random_mean = torch.mean(x_random, keepdim=True, dim=(2, 3))
        x_std = torch.std(x, keepdim=True, dim=(2, 3))
        x_random_std = torch.std(x_random, keepdim=True, dim=(2, 3))
        gamma = np.random.uniform(0, 1)

        # get style-random (SR) features
        mix_mean = gamma * x_mean + (1 - gamma) * x_random_mean
        mix_std = gamma * x_std + (1 - gamma) * x_random_std
        SR_rep = (x - x_mean) / (x_std+1e-5) * mix_std + mix_mean

        # get content-random (CR) features
        CR_rep = (x_random - x_random_mean) / (x_random_std+1e-5) * x_std + x_mean

        if self.dataset == "sleep":
            return self.sleep_post_layers(SR_rep, CR_rep)
        else:
            return

    def forward(self, x):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out

class ProxyPLoss(nn.Module):
	'''
    borrowed from here
	https://github.com/yaoxufeng/PCL-Proxy-based-Contrastive-Learning-for-Domain-Generalization
	'''
	def __init__(self, num_classes, scale):
		super(ProxyPLoss, self).__init__()
		self.soft_plus = nn.Softplus()
		self.label = torch.LongTensor([i for i in range(num_classes)])
		self.scale = scale
	
	def forward(self, feature, target, proxy):
		feature = F.normalize(feature, p=2, dim=1)
		pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)
		label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0))  # (C, N)
		pred = torch.masked_select(pred.transpose(1, 0), label)  # N,
		
		pred = pred.unsqueeze(1)  # (N, 1)
		
		feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
		label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)
		
		index_label = torch.LongTensor([i for i in range(feature.shape[0])])  # generate index label
		index_matrix = index_label.unsqueeze(1) == index_label.unsqueeze(0)  # get index matrix
		
		feature = feature * ~label_matrix  # get negative matrix
		feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)
		
		logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
		label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
		loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)
		
		return loss

class PCL(Base):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super().__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        if dataset == "sleep":
            self.head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            self.g_net = nn.Parameter(torch.randn(5, 128))

    def forward_train(self, x):
        x = self.feature_cnn(x)
        out = F.normalize(x, p=2, dim=1) @ F.normalize(self.g_net, p=2, dim=1).T

        x_rep = self.head(x)
        w_rep = self.head(self.g_net)
        return out, x_rep, w_rep
    
    def forward(self, x):
        x = self.feature_cnn(x)
        out = F.normalize(x, p=2, dim=1) @ F.normalize(self.g_net, p=2, dim=1).T
        return out

class MLDG(Base):
    def __init__(self, dataset, device=None, voc_size=None, model=None, ehr_adj=None, ddi_adj=None):
        super().__init__(dataset, device, voc_size, model, ehr_adj, ddi_adj)
        self.dataset = dataset
    def forward(self, x):
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return out
    
    def functional_forward(self, x, fast_weights):
        x = self.feature_cnn.functional_forward(x, fast_weights)
        if self.dataset == "sleep":
            out = F.linear(x, fast_weights['g_net.0.weight'], fast_weights['g_net.0.bias'])
            out = F.relu(out)
            out = F.linear(out, fast_weights['g_net.2.weight'], fast_weights['g_net.2.bias'])

        return out