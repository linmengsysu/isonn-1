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
        self.valid_loader = test_loader#data_loader[1]
        self.num_train = len(self.train_loader.sampler.indices)
        self.num_valid = 0#len(self.valid_loader.sampler.indices)
     
        
        self.test_loader = test_loader
        self.num_test = len(self.test_loader.dataset)
        print('self.num_train', self.num_train, 'self.num_test',self.num_test)
        self.device = torch.device("cuda:0")
        self.counter = 0
        self.epochs = 100#config.epochs
        self.start_epoch = 0
       
        self.lr = config.init_lr
        self.best_valid_acc = 0
        self.best_valid_f1 = 0
        self.train_patience = config.train_patience

        

        self.plot_dir = '../../result/expIsoNN/ADHD/fold_' + str(self.fold_count) + '_dropout_' + str(self.dropout) + '_k_' + str(self.k) + '_c_' + str(self.c) + '_epoch_' + str(self.epochs)
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


            # self.model.eval()
            # valid_acc, valid_f1 = self.validate(epoch)
            # valid_acc = 0
            # valid_f1 = 0
            # result = self.test()
            
            # scores.append([valid_acc, valid_f1])
            # # msg = "train loss: {:.3f} - train acc: {:.3f} "
            # # print(msg.format(train_loss, train_acc))
            # is_best = valid_acc > self.best_valid_acc
            # msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            # msg2 = "- val acc: {:.3f} - val f1: {:.3f}"
            # if is_best:
            #     self.counter = 0
            #     msg2 += " [*]"
            # msg = msg1 + msg2
            # print(msg.format(train_loss, train_acc, valid_acc, valid_f1))

            # # check for improvement
            # if not is_best:
            #     self.counter += 1
            # if self.counter > self.train_patience:
            #     print("[!] No improvement in a while, stopping training.")
            #     break
            # self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            # self.save_checkpoint(
            #     {'epoch': epoch + 1,
            #      'model_state': self.model.state_dict(),
            #      'optim_state': self.optimizer.state_dict(),
            #      'best_valid_acc': self.best_valid_acc,
            #      }, is_best
            # )

            
        # elapsed = time.time() - start
        # scores = np.array(scores)
        # train_scores = np.array(train_scores)
        # plt.figure()
        # plt.plot(np.arange(len(train_scores)), train_scores[:,0], 'o--', color='tab:blue', linewidth=2, label='loss')
        
        # plt.tick_params(axis='y', labelsize=14)
        # plt.tick_params(axis='x', labelsize=14)
        # plt.ylabel('Loss', fontsize=16)
        # plt.xlabel('Epoch Number', fontsize=16)
        # plt.legend(prop={'size': 14})
        # plt.savefig(self.plot_dir + '_loss.pdf')
        # plt.clf()
        
        # plt.plot(np.arange(len(scores)), scores[:,0], 'o--', color='tab:blue', linewidth=2, label='Acc')
        # plt.plot(np.arange(len(scores)), scores[:,1],'*--', color='tab:orange', linewidth=2, label='F1')
        # plt.plot(np.arange(len(scores)), train_scores[:,1],'v--', color='tab:green', linewidth=2, label='TrainAcc')
       

        # plt.tick_params(axis='y', labelsize=14)
        # plt.tick_params(axis='x', labelsize=14)
        # # plt.ylabel('Loss', fontsize=16)
        # plt.xlabel('Epoch Number', fontsize=16)
        # plt.legend(prop={'size': 14})
        # plt.savefig(self.plot_dir + '_test_performance.pdf')
        # plt.clf()

        # f = open(self.plot_dir + '_all_scores', 'wb')
        # ss = {'train-loss-acc': train_scores, 'test-acc-f1': scores}
        # pickle.dump(ss,f)
        # f.close()

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
        aucs = AverageMeter()
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



    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        correct = 0
        losses = AverageMeter()
        accs = AverageMeter()
        f1s = AverageMeter()
        recalls = AverageMeter()
        aucs = AverageMeter()
        precisions = AverageMeter()
        self.model.eval()
      
        for i, (x, y) in enumerate(self.valid_loader):
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

            loss = F.nll_loss(probas, y)

            acc = accuracy_score(y.cpu(), predicted.cpu())
            f1 = f1_score(y.cpu(), predicted.cpu())
            prec = precision_score(y.cpu(), predicted.cpu())
            rec = recall_score(y.cpu(), predicted.cpu())


            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc, x.size()[0])
            f1s.update(f1, x.size()[0],)
            recalls.update(rec, x.size()[0])
            precisions.update(prec, x.size()[0])
                
              

        print('k, c, dropout', self.k, self.c, self.dropout)
        print('Accuracy:', accs.avg)
        print('F1: ', f1s.avg)
        print('Precision: ', precisions.avg)
        print('Recall: ', recalls.avg)

        # result = [accs.avg, f1s.avg, precisions.avg, recalls.avg]
        return accs.avg, f1s.avg

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
        aucs = AverageMeter()
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

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )


    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        print(filename)
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )


