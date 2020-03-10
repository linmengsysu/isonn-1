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
        # self.fc1 = nn.Linear(self.nodes[0] ** 2, self.nodes[1] ** 2)
        # self.fc2 = nn.Linear(self.nodes[0] ** 2, self.nodes[2] ** 2)
        # self.fc3 = nn.Linear(self.nodes[0] ** 2, self.nodes[3] ** 2)


    def forward(self, x, last=True):
        self.e += 1
        kernels = []
        for i in range(len(self.feature_networks)):
            x, kernel = self.feature_networks[i](x)
            kernels.append(kernel.detach().cpu())
            # plt.clf()
            # plt.hist(x.detach().cpu().numpy().reshape(-1)) 
            # plt.title("Histogram with 'auto' bins")
            # plt.savefig('plots/softmax_layer_{}_feature_historgram.pdf'.format(i))

        # print('x.size()', x.size(), x.view(-1, self.nodes[0] ** 2).size())
        # res1 = self.fc1(x.view(-1, self.nodes[0] ** 2)).view(-1, 1, self.nodes[1], self.nodes[1])
        # x2, kernel2 = self.feature_networks[1](x1+res1)

        # res2 = self.fc2(x.view(-1, self.nodes[0] ** 2)).view(-1, 1, self.nodes[2], self.nodes[2])
        # x, kernel3 = self.feature_networks[2](x2+res2)

        # kernels = [kernel1.detach().cpu(), kernel2.detach().cpu(), kernel3.detach().cpu()]
        # f = open('ADHD_3_layer_k3c3_kernels_no_res','wb')
        # pickle.dump(kernel.detach().cpu(), f)
        # f.close()
        # res3 = self.fc3(x.view(-1, self.nodes[0] ** 2)).view(-1, 1, self.nodes[3], self.nodes[3])
        # x = self.feature_networks[3](x3+res3)

        x = x.contiguous().view(-1, self.nfeatures) #+ res
        # plt.clf()
        # plt.hist(x.detach().numpy().reshape(-1)) 
        # plt.title("Histogram with 'auto' bins")
        # plt.savefig('plots/epoch_{}_feature_historgram.pdf'.format(self.e))

        prob = self.classifier(x)
        return prob

  