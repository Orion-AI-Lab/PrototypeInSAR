import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import math
import torch.utils.data as data_utils
import torch.nn.functional as F
from Models import *
from train_utils import*
import torch.utils.data as utils
import pickle
import argparse
import Dataset
import torchsummary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial_learning_rate')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--h', type=int, default=3, help='dimension of the hidden layer')
    parser.add_argument('--scale', type=float, default=2, help='scaling factor for distance')
    parser.add_argument('--reg', type=float, default=1, help='regularization coefficient')
    parser.add_argument('--pretrained_imagenet', type=bool, default=True)
    parser.add_argument('--l1', type=float, default=1, help='regularization coefficient')
    parser.add_argument('--epochs', type=float, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--stepLR', type=bool, default=False)
    parser.add_argument('--adapt_weight', type=int, default=1)
    parser.add_argument('--cosine', type=bool, default=False)
    parser.add_argument('--synthetic_train_dataset', type=str,default='')
    parser.add_argument('--synthetic_val_dataset', type=str,default='')
    parser.add_argument('--test_dataset', type=str,default='')
    parser.add_argument('--encoder', type=str,default='swin')



    args, _ = parser.parse_known_args()


    data_dir = args.synthetic_train_dataset
    test_dir = args.test_dataset
    val_dir = args.synthetic_val_dataset

    if data_dir=='' or test_dir=='' or val_dir=='':
        print('Give proper data paths')
        exit(2)


    train_dataset = Dataset.Dataset(data_dir, setting='train', sim=False,original=False)
    full_dat = train_dataset

    val_set = Dataset.Dataset(val_dir, setting='test', original=False, sim=False)

    test_dataset =Dataset.Dataset(test_dir, setting='test', original=False, sim=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=44,drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    loader = [test_loader,validation_loader]
    dataset_train_len=len(train_dataset)


    model = NetXd(args.h,args.num_classes,args.scale,encoder=args.encoder)
    device='cuda:0'
    model = model.to(device)

    optimizer_s = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    train_model(model, optimizer_s,args.learning_rate,args.epochs,args.reg, train_loader=train_loader,test_loader=loader,dataset_train_len=dataset_train_len,l1=args.l1,args=args)
