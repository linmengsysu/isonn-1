import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import math 
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt 

from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment

import random
import pickle

class Isomorphic_Feature_Extraction(nn.Module):
    """docstring for ClassName"""
    def __init__(self, k, c_prev, c, nodes, n_hid, dropout):
        super(Isomorphic_Feature_Extraction, self).__init__()
        self.k = k
        self.c = c
        self.lc = int(self.c / c_prev)
        self.fac_k = math.factorial(k)
        self.P = self.get_all_P(self.k)
        self.dropout = dropout
        self.device = torch.device("cuda:0")
        self.kernel = nn.Parameter(torch.randn(self.lc, k, k))
        self.maxpoolk = nn.MaxPool2d((1, self.fac_k), stride=(1, self.fac_k))
        self.avgpool = nn.AvgPool2d((2, 2), stride=(1, 1))
        self.n_subgraph = (nodes - self.k + 1) ** 2 
        self.n_input = self.n_subgraph
        self.e = 0
        
        self.biases = nn.Parameter(torch.randn(self.c))
        # self.fc = nn.Linear(n_input, n_hid)
        

    def forward(self, x):
        self.e += 1
        # layer = self.FastIsoLayerWithP(x, self.kernel)
        layer = self.IsoLayer(x, self.kernel) # [B, n_subgraph, c]
        maximum = torch.max(layer, dim=2, keepdim=True)[0]
        minimum = torch.min(layer, dim=2, keepdim=True)[0]
        # print('minimum, maximum', minimum[0], maximum.size())
        # plt.clf()
        # plt.hist(layer.cpu().detach().numpy().reshape(-1)) 
        # plt.title("Histogram with 'auto' bins")
        # plt.savefig('plots/layer_historgram.pdf')

        H = int(np.sqrt(self.n_subgraph))
        # layer = (1 - (layer - minimum)/(maximum - minimum)).transpose(2,1) # [B, c, n_subgraph]  
        if self.c >= 1:
            layer =  F.softmax(-layer, dim=1).transpose(2,1)
            print('subgraph softmax')
        else:
            layer =  F.softmax(-layer, dim=2).transpose(2,1)
            # print('channel softmax')
        # print('layer', layer.size())

        layer = layer.view(-1, self.c, H, H)
        # print('self.kernel', self.kernel.detach().cpu())
        # f = open('kernel','wb')
        # pickle.dump(self.kernel.detach().cpu(), f)
        # f.close()
        # layer = self.avgpool(layer)
        # print('avg pool layer', layer.size())
        plt.clf()
        plt.hist(layer.cpu().detach().numpy().reshape(-1)) 
        plt.title("softmax subgraphs Histogram with 'auto' bins")
        plt.savefig('plots/subgraph_normalized_layer_historgram.pdf')
        return layer, self.kernel

    def IsoLayer(self, x, kernel):
        print('IsoLayer')
        x = self.get_all_subgraphs(x, self.k)
        B, n_subgraph, c_prev,  k, k = x.size()
        
        # print(self.lc, x.size())
        x = x.view(-1, 1, self.k, self.k)

        tmp = torch.matmul(torch.matmul(self.P, kernel.view(self.lc, 1, self.k, self.k)), torch.transpose(self.P, 2, 1)).view(-1, self.k, self.k) - x #[B*n_subgraph, 1, k, k] - [c*k!, k, k]
        raw_features = -1 * torch.norm(tmp, p='fro', dim=(-2,-1)) ** 2
        raw_features = raw_features.view(B, n_subgraph, self.c, self.fac_k)
        
        feature_P = self.maxpoolk(raw_features).view(B, -1, self.c)
        feature_P = (-1) * feature_P

        return feature_P


    def FastIsoLayerWithP(self, x, kernel):
        print('fast IsoLayer')
        x = self.get_all_subgraphs(x, self.k)
        
        B, n_subgraph, c_prev, k, k = x.size()
        x = x.view(-1, self.k, self.k)
        P = self.compute_p(x, kernel)# P [B*n_subgraph, c, k, k] 
        # print('P size', P.size())
        x = x.view(-1, 1, self.k, self.k)
        # print('x size', x.size(),torch.matmul(P, kernel).size(),torch.transpose(P, 2, 1).size())
        # print(torch.matmul(torch.matmul(P, kernel), torch.transpose(P, 2, 1)).size())
        tmp = torch.matmul(torch.matmul(P, kernel), torch.transpose(P, 3, 2)) - x #[B*n_subgraph, 1, k, k] - [B*n_subgraph, c, k, k] 
        # print('tmp size', tmp.size())
        features = torch.norm(tmp, p='fro', dim=(-2,-1)) ** 2
        features = features.view(B, n_subgraph, self.c)
        return features
  
    def compute_p(self, subgraphs, kernel): 
        c, k, k = kernel.size()
        N, k, k = subgraphs.size() # N = B * n_subgraph
        if torch.cuda.is_available():
            VGs, UGs = LA.eig(subgraphs.detach().cpu().numpy()) 
            VHs, UHs = LA.eig(kernel.detach().cpu().numpy())
        else:   
            VGs, UGs = LA.eig(subgraphs.detach().numpy()) 
            VHs, UHs = LA.eig(kernel.detach().numpy())

        bar_UGs = np.absolute(UGs).reshape(-1, 1, k, k)
        bar_UHs = np.absolute(UHs)

        P = np.matmul(bar_UGs, np.transpose(bar_UHs,(0,2,1)))
        P_star = torch.from_numpy(np.array(P)).requires_grad_(False)
        if torch.cuda.is_available():
            P_star = P_star.type(torch.cuda.FloatTensor)
        else:
            P_star = P_star.type(torch.FloatTensor)
        return P_star
        

    # get all possible P (slow algo)
    def get_all_P(self, k):
        n_P = np.math.factorial(k)
        P_collection = np.zeros([n_P, k, k])
        perms = permutations(range(k), k)

        count = 0
        for p in perms:
            for i in range(len(p)):
                P_collection[count, i, p[i]] = 1
            count += 1
        Ps = torch.from_numpy(np.array(P_collection)).requires_grad_(False)
        if torch.cuda.is_available():
            Ps = Ps.type(torch.cuda.FloatTensor)
        else:
            Ps = Ps.type(torch.FloatTensor)
        # Ps = Ps.to(self.device)
        return Ps

    def get_all_subgraphs(self, X, k):
        # X = X.detach().squeeze()
        (batch_size, c, n_H_prev, n_W_prev) = X.size()

        n_H = n_H_prev - k + 1
        n_W = n_W_prev - k + 1
        subgraphs = []
        for h in range(n_H):
            for w in range(n_W):
                x_slice = X[:, :, h:h+k, w:w+k]
                subgraphs.append(x_slice)
        S = torch.stack(subgraphs, dim=1) # [B, n_subgraph, c, k, k]
        # print('S.size()', S.size())
        return S



class Classification_Component(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, nclass, dropout):
        super(Classification_Component, self).__init__()
        self.input_size = input_size
        self.dropout = dropout

        self.fc1 = nn.Linear(input_size, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, nclass)

    def forward(self, features):
        h1 = F.relu(self.fc1(features))
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, self.dropout, training=self.training)
        pred = F.log_softmax(self.fc3(h2), dim=1)
        return pred







