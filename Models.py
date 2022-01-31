
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torchvision
import timm
from torch.autograd import Function
import numpy as np


class NetXd(nn.Module):
    def __init__(self, num_hidden_units=2, num_classes=10,s=2):
        super(NetXd, self).__init__()
        self.scale=s
        self.encoder =timm.create_model('swin_base_patch4_window7_224',pretrained=True)
        print(self.encoder)
        self.encoder.head = nn.Identity()

        self.fc = nn.Linear(1024,num_hidden_units)
        self.dce= dce_loss(num_classes,num_hidden_units)

    def forward(self, x,domain=False):

        x = self.encoder(x)
        x1 = self.fc(x)
        centers,x=self.dce(x1)
        output = F.log_softmax(self.scale*x, dim=1)
        return x1,centers,x,output


class dce_loss(torch.nn.Module):
    def __init__(self, n_classes,feat_dim,init_weight=True):

        super(dce_loss, self).__init__()
        device=0
        self.n_classes=n_classes
        self.feat_dim=feat_dim
        self.centers=nn.Parameter(torch.randn(self.feat_dim,self.n_classes).cuda(device),requires_grad=True)
        print('Centers : ',self.centers.shape)

        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

     

    def forward(self, x):

        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers,2),0, keepdim=True)
        features_into_centers=2*torch.matmul(x, (self.centers))
        dist=features_square+centers_square-features_into_centers

        return self.centers, -dist

def regularization(features, centers, labels):
        #print('features : ' , features.shape)
        #print('centers : ',torch.t(centers)[labels].shape)
        distance=(features-torch.t(centers)[labels])

        distance=torch.sum(torch.pow(distance,2),1, keepdim=True)

        distance=(torch.sum(distance, 0, keepdim=True))/features.shape[0]

        return distance