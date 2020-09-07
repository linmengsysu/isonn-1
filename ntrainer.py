import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable
import torch.optim as optim

import os
import time
import shutil
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt 
import pickle

from tqdm import tqdm
from utils import AverageMeter
from nmodel import IsoNN
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_score, recall_score

NVIDIA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=2

class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, fold, data_loader, test_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        - k: kernel size, array
        - c: channel number, array
        """ 
        self.result_dir = config.result_dir
        self.config = config
        self.fold_count = fold
        self.dropout = config.dropout
        self.cuda = config.cuda
        # self.node = config.num_node

        self.k = [4]
        self.c = [3]
        self.num_node = config.num_node
        # self.feature_size = ((config.num_node - self.k[0]) + 1 - self.k[1] + 1) ** 2 * self.c[0] * self.c[1]
        # self.feature_size = ((config.num_node - self.k) + 1 - self.k + 1) ** 2 * self.c
        # self.feature_size = self.feature_size1 + self.feature_size2
        self.n_hidden1 = config.hidden1
        self.n_hidden2 = config.hidden2
        self.nclass = config.nclass
       
        self.batch_size = config.batch_size
        self.M = config.M
        self.best = False#False #True
        
        self.train_loader = data_loader[0]
        # self.valid_loader = test_loader
        self.num_train = len(self.train_loader.sampler.indices)

        self.test_loader = test_loader
        self.num_test = len(self.test_loader.dataset)
        print('self.num_train', self.num_train, 'self.num_test',self.num_test)
        self.device = torch.device("cuda:0")
        self.counter = 0
        self.epochs = 100#config.epochs
        self.start_epoch = 0
       
        self.lr = config.init_lr
        # self.best_valid_acc = 0
        # self.best_valid_f1 = 0
        self.train_patience = config.train_patience

        # self.plot_dir = '../../result/expIsoNN/ADHD/fold_' + str(self.fold_count) + '_dropout_' + str(self.dropout) + '_k_' + str(self.k) + '_c_' + str(self.c) + '_epoch_' + str(self.epochs)
        self.ckpt_dir = config.ckpt_dir

        self.model_name = 'IsoNN_k_{}_c_{}'.format(
            self.k, self.c
        )

        # build RAM model
        self.model = IsoNN(self.k, self.c, self.num_node, self.n_hidden1, self.n_hidden2, self.nclass, self.dropout)
        if self.cuda:
            # print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model.cuda()
            # self.model = torch.nn.DataParallel(self.model)

        # self.model = self.model.to(self.device)
        #Adam
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-3, #weight_decay=1e-1
        )


    def train(self):
        train_scores = []
        scores = []
        start = time.time()
        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )
            # train for 1 epoch
            self.model.train()
            train_loss, train_acc = self.train_one_epoch(epoch)
            train_scores.append([train_loss, train_acc])

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        f1s = AverageMeter()
        recalls = AverageMeter()
        precisions = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader): # x [B, W, H], y [B]
                torch.cuda.empty_cache()
                x, y = Variable(x), Variable(y)
                if self.cuda:
                    x, y = x.cuda().to(self.device), y.cuda().to(self.device)
                probas = self.model(x)
                # print("probas", probas.size())
                predicted = torch.max(probas, 1)[1]

                loss = F.nll_loss(probas, y) # supervised loss
                acc = accuracy_score(y.cpu(), predicted.cpu())
                f1 = f1_score(y.cpu(), predicted.cpu())
                prec = precision_score(y.cpu(), predicted.cpu())
                rec = recall_score(y.cpu(), predicted.cpu())
                
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc)
                f1s.update(f1)
                recalls.update(rec)
                precisions.update(prec)
                
                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)
                pbar.update(self.batch_size)
        return losses.avg, accs.avg

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        # self.load_checkpoint(best=self.best)
        correct = 0
        accs = AverageMeter()
        f1s = AverageMeter()
        recalls = AverageMeter()
        precisions = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = Variable(x), Variable(y)
                if self.cuda:
                    x, y = x.cuda(), y.cuda()
                # duplicate 10 times
                x = x.repeat(self.M, 1, 1, 1)
                probas = self.model(x)
                
                probas = probas.view(
                    self.M, -1, probas.shape[-1]
                )
                probas = torch.mean(probas, dim=0)
                predicted = probas.data.max(1, keepdim=True)[1]

                acc = accuracy_score(y.cpu(), predicted.cpu())
                f1 = f1_score(y.cpu(), predicted.cpu())
                prec = precision_score(y.cpu(), predicted.cpu())
                rec = recall_score(y.cpu(), predicted.cpu())
                
                accs.update(acc)
                f1s.update(f1)
                recalls.update(rec)
                precisions.update(prec)

        print('k, c, dropout', self.k, self.c, self.dropout)
        print('Accuracy:', accs.avg)
        print('F1: ', f1s.avg)
        print('Precision: ', precisions.avg)
        print('Recall: ', recalls.avg)

        result = [accs.avg, f1s.avg, precisions.avg, recalls.avg]
        return result

