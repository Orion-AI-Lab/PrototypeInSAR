import random

import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.misc
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import torchvision
import numpy as np
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch.utils.data import *
from Models import *
import Models
import Dataset
import matplotlib.pyplot as plt
import os
import cv2 as cv
import kornia
import shutil


def train_model(cnn, optimizer_s, lrate, num_epochs, reg, train_loader, test_loader, dataset_train_len, l1=1, args=None):
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    criterion = nn.CrossEntropyLoss()
    device = 'cuda:0'
    cnn.cuda(device)

    epochs = []
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    train_error = []
    test_error = []
    best_acc = 0.0
    total = num_epochs * len(train_loader.dataset) / args.batch_size
    best_val = 0.0
    best_test = 0.0
    best_test_acc = 0.0
    current_test_acc = 0.0
    best_stats = {}
    current_stats = {}
    val_loss_list = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=len(train_loader), eta_min=0,
                                                                last_epoch=-1)
    for epoch in range(num_epochs):

        cnn.train()
        epochs.append(epoch)
        optimizer = optimizer_s
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('*' * 70)
        running_loss = 0.0
        running_corrects = 0.0
        train_batch_ctr = 0.0
        for i, ((image, _,), label) in enumerate(train_loader):
            if i%100==0:
                print('learning_rate ', scheduler.get_lr()[0])
            Models.p = ((i + 1) + (len(train_loader.dataset) / args.batch_size) * (epoch)) / total
            image = image.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()

            features, centers, distance, outputs = cnn(image)
            _, preds = torch.max(distance, 1)

            loss1 = F.nll_loss(outputs, label)
            loss2 = regularization(features, centers, label)
            deforms = features[label==1]
            normal = features[label==0]
            prototype_1 = centers[0,:]
            prototype_2 = centers[1,:]

            prototype_distance = nn.PairwiseDistance(p=2)(prototype_1,prototype_2)
            loss_proto_distance = - prototype_distance
            loss = l1 * loss1 + reg *(loss2)
            if i % 100 == 0:
                print('Epoch ', epoch, ' Iteration ', i, 'Loss ', loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_batch_ctr = train_batch_ctr + 1

            running_loss += loss.item()

            running_corrects += torch.sum(preds == label.data)

            epoch_acc = (float(running_corrects) / (float(dataset_train_len)))

        print('Train corrects: {} Train samples: {} Train accuracy: {}'.format(running_corrects, (dataset_train_len),
                                                                               epoch_acc))
        train_acc.append(epoch_acc)
        train_loss.append(1.0 * running_loss / train_batch_ctr)
        train_error.append(((dataset_train_len) - running_corrects) / (dataset_train_len))

        cnn.eval()

        for idx, loader in enumerate(test_loader):
            test_running_corrects = 0.0
            test_batch_ctr = 0.0
            test_running_loss = 0.0
            test_total = 0.0

            false_positives = 0
            total_negatives = 0
            false_negatives = 0
            true_positives = 0
            true_negatives = 0
            for (image, _), label in loader:
                with torch.no_grad():
                    image = image.cuda(device)
                    label = label.cuda(device)
                    features, centers, distance, test_outputs = cnn(image)

                    _, predicted_test = torch.max(distance, 1)

                    loss1 = F.nll_loss(test_outputs, label)
                    loss2 = regularization(features, centers, label)

                    loss = loss1 + reg * loss2

                    test_running_loss += loss.item()
                    test_batch_ctr = test_batch_ctr + 1

                    test_running_corrects += torch.sum(predicted_test == label.data)
                    test_epoch_acc = (float(test_running_corrects) / float(len(loader.dataset)))

                    total_negatives += (predicted_test == 0).sum().item()
                    false_positives += (predicted_test[label == 0] == 1).sum().item()
                    false_negatives += (predicted_test[label == 1] == 0).sum().item()
                    true_positives += (predicted_test[label == 1] == 1).sum().item()
                    true_negatives += (predicted_test[label == 0] == 0).sum().item()
            test_acc.append(test_epoch_acc)
            test_loss.append(1.0 * test_running_loss / test_batch_ctr)
            if idx == 0:
                current_test_acc = test_epoch_acc
                current_stats = {
                    'FP': false_positives,
                    'FN': false_negatives,
                    'TP': true_positives,
                    'TN': true_negatives,
                    'Epoch': epoch + 1
                }
            if idx == 1:
                val_loss_list.append(test_epoch_acc)
                plt.plot(range(len(val_loss_list)),val_loss_list)
                plt.title('Validation Loss')
                plt.show()
                if test_epoch_acc > best_val:
                    best_val = test_epoch_acc
                    best_test_acc = current_test_acc
                    best_stats = current_stats
                    torch.save(cnn, 'best_model.pt')
                    print('New best validation score : ', test_epoch_acc)
                    print('C1 Acc : ', best_test_acc)
                    print('Best Stats : ', best_stats)
                    torch.save(cnn, 'bestmodels.pt')

            print(
                'Test corrects: {} Test samples: {} Test accuracy {}'.format(test_running_corrects,
                                                                             (len(loader.dataset)),
                                                                             test_epoch_acc))

            print('Train loss: {} Test loss: {}'.format(train_loss[epoch], test_loss[epoch]))
            print('Test Running corrects : ', test_running_corrects)

            print('*' * 70)

            if idx == 0:
                tmp_set = 'C1 Classification '
            elif idx == 1:
                tmp_set = 'Synth Test Classification'
            else:
                tmp_set = 'Validation Test Classification'

            print('Task: ',tmp_set)
            print('='*20)
            print('False Positives : ', false_positives)
            print('True Positives: ', true_positives)
            print('False Negatives : ', false_negatives)
            print('True Negatives : ', true_negatives)
            print('*' * 70)

        test_acc.append(test_epoch_acc)
        test_loss.append(1.0 * test_running_loss / test_batch_ctr)

    print('Train ended \n\n\n')
    print('Best test accuracy : ', best_test_acc)
    print('Best Stats : ', best_stats)