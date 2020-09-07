import math
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn

from torch.distributions import Normal

from nmodules import Isomorphic_Feature_Extraction, Classification_Component
import pickle

class IsoNN(nn.Module):
  
    def __init__(self, k, c, node, nhid1, nhid2, nclass, dropout):
     
        super(IsoNN, self).__init__()
        self.nodes = [node]
        self.channels = [1]
        for i in range(len(k)):
            self.nodes.append(self.nodes[i]-k[i]+1) 
            self.channels.append(c[i])
        # self.c = c
        self.k = k
        # print('num of nodes/channels at each layer', self.nodes, self.channels)
        self.feature_networks = [Isomorphic_Feature_Extraction(self.k[i], self.channels[i], self.channels[i+1], self.nodes[i], nhid1, dropout) for i in range(len(self.k))]
        for i, feature in enumerate(self.feature_networks):
            self.add_module('features_{}'.format(i), feature)

        self.nfeatures = self.nodes[-1] ** 2 * self.channels[-1]
        # for i in range(len(c)):
        #     self.nfeatures *= c[i]
        self.classifier = Classification_Component(self.nfeatures, nhid1, nhid2, nclass, dropout)
        self.e = 0

    def forward(self, x, last=True):
        self.e += 1
        kernels = []
        for i in range(len(self.feature_networks)):
            x, kernel = self.feature_networks[i](x)
            kernels.append(kernel.detach().cpu())

        x = x.contiguous().view(-1, self.nfeatures)
        prob = self.classifier(x)
        return prob

  
