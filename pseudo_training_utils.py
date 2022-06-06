import argparse
import torch
from torch import optim
import torch.nn.functional as F
import Models
import Dataset
import cv2 as cv
import torch.nn as nn
import os
import glob


def save_image(label, image, idx, target_set):
    image = image.cpu().squeeze().permute(1, 2, 0).numpy()

    path = target_set + str(label.cpu().item()) + '/' + str(idx) + '.png'
    status = cv.imwrite(path, image)
    if status==False:
        print('Sample:', idx, ' Writing failed \n Exiting')
        exit(2)


def generate_pseudo(models=None, original_set='unlabeled/',
                    target_set='pseudo/', append=False,arch='swin',model_root_path='models/'):
    print('='*20)
    print('Generating pseudo labels using: ',arch,' model')

    if arch == 'swin':
        arch_path = 'swin/swin.pt'
    elif arch == 'deit':
        arch_path = 'deit/deit.pt'
    elif arch == 'convit':
        arch_path = 'convit/convit.pt'


    if models is None:
        model_path = model_root_path + arch_path
        print('Checkpoint path: ', model_path)
        models = torch.load(model_path, map_location='cuda:0')

    print('=' * 20)
    models.eval()
    pseudo_dir = target_set
    files = glob.glob(pseudo_dir + '/1/*')
    if append == False:
        for f in files:
            os.remove(f)
        files = glob.glob(pseudo_dir + '/0/*')
        for f in files:
            os.remove(f)
    test_dir = original_set
    test_dataset = Dataset.Unlabeled(test_dir, setting='test', original=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    device = 'cuda:0'
    counter = 0
    for (image, original), _ in test_loader:
        image = image.to(device)

        list_of_preds = []
        features, centers, distance, test_outputs = models(image)
        _, predicted_test = torch.max(distance, 1)
        list_of_preds.append(predicted_test.cpu().numpy())
        predictions = predicted_test
        save_image(predictions, original, counter, target_set)
        counter += 1


def pseudo_train(arch='swin',pseudo_train_dir = '', test_dir = '', model_root_path='models/',  synthetic_val_dir = ''):
    if arch == 'swin':
        arch_path = 'swin/swin.pt'
    elif arch == 'deit':
        arch_path = 'deit/deit.pt'
    elif arch == 'convit':
        arch_path = 'convit/convit.pt'

    model_paths = model_root_path + arch_path
    model = torch.load(model_paths, map_location='cuda:0')

    test_dataset = Dataset.Dataset(test_dir, setting='test', original=False,
                                   sim=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=120, shuffle=False, num_workers=1,
                                              drop_last=False)
    device = 'cuda:0'
    synth_dataset = Dataset.Dataset(synthetic_val_dir, setting='test', original=False,
                                    sim=False)

    synth_loader = torch.utils.data.DataLoader(synth_dataset, batch_size=40, shuffle=False, num_workers=1)
    best_synth = 0.0
    best_score = 0.0
    best_stats = {}
    print('='*20)
    print('Begin Pseudo Training')
    print('=' * 20)
    train_dataset = Dataset.Dataset(pseudo_train_dir, setting='train', original=False,
                                    sim=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=1)
    for param in model.encoder.parameters():
        param.requires_grad = False
    a = [8096, 16000]

    w = 1
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, a[0] * w), nn.ReLU(), nn.Linear(a[0] * w, a[1] * w), nn.ReLU(),
                             nn.Linear(a[1] * w,
                                       3))
    model.dce.centers.requires_grad = False
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    model.to(device)
    for epoch in range(10):
        model.train()
        for i, ((image, _,), label) in enumerate(train_loader):
            image = image.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            features, centers, distance, outputs = model(image)
            _, preds = torch.max(distance, 1)
            loss1 = F.nll_loss(outputs, label)
            loss2 = Models.regularization(features, centers, label)
            loss = loss1 + (loss2)
            if i % 100 == 0:
                print('Epoch ', epoch, ' Iteration ', i, 'Loss ', loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

        test_corrects = 0.0
        total_negatives = 0.0
        false_positives = 0.0
        true_positives = 0.0
        false_negatives = 0.0
        true_negatives = 0.0
        device = 'cuda:0'
        counter = 0
        model.eval()
        for (image, original), label in test_loader:
            image = image.to(device)
            label = label.to(device)
            predictions = torch.zeros(label.shape)
            list_of_preds = []

            features, centers, distance, test_outputs = model(image)
            _, predicted_test = torch.max(distance, 1)
            list_of_preds.append(predicted_test.cpu().numpy())
            predictions = predicted_test
            counter += 1

            total_negatives += (predicted_test == 0).sum().item()
            false_positives += (predictions[label == 0] == 1).sum().item()
            false_negatives += (predictions[label == 1] == 0).sum().item()
            true_positives += (predictions[label == 1] == 1).sum().item()
            true_negatives += (predictions[label == 0] == 0).sum().item()
            test_corrects += torch.sum(predictions == label.data)
        current_stats = {
            'FP': false_positives,
            'FN': false_negatives,
            'TP': true_positives,
            'TN': true_negatives,
        }

        # Evaluate on validation set
        synth_test_corrects = 0.0
        synth_total_negatives = 0.0
        synth_false_positives = 0.0
        synth_true_positives = 0.0
        synth_false_negatives = 0.0
        synth_true_negatives = 0.0
        for (image, original), label in synth_loader:
            image = image.to(device)
            label = label.to(device)
            predictions = torch.zeros(label.shape)
            list_of_preds = []

            features, centers, distance, test_outputs = model(image)
            _, predicted_test = torch.max(distance, 1)
            list_of_preds.append(predicted_test.cpu().numpy())
            predictions = predicted_test  # + predictions
            counter += 1

            synth_total_negatives += (predicted_test == 0).sum().item()
            synth_false_positives += (predictions[label == 0] == 1).sum().item()
            synth_false_negatives += (predictions[label == 1] == 0).sum().item()
            synth_true_positives += (predictions[label == 1] == 1).sum().item()
            synth_true_negatives += (predictions[label == 0] == 0).sum().item()
            synth_test_corrects += torch.sum(predictions == label.data)
        current_synth = synth_test_corrects / len(synth_loader.dataset)
        if current_synth > best_synth:
            best_synth = current_synth
            best_score = test_corrects.cpu().numpy() / len(test_loader.dataset)
            best_stats = current_stats
            print('New Best Validation score :', best_synth)
            print('New Best model accuracy : ', best_score)
            print('Stats : ', best_stats)
            torch.save(model, arch + 'PLPseudo.pt')

    print('New Best Validation score :', best_synth)
    print('New Best model accuracy : ', best_score)
    print('Stats : ', best_stats)

def generate_and_train(unlabeled_set,target_path,model_root_path,arch,synthetic_validation_path,test_path):
    generate_pseudo(original_set=unlabeled_set,
                    model_root_path=model_root_path, arch=arch)
    pseudo_train(arch=arch,
                 synthetic_val_dir=synthetic_validation_path,
                 model_root_path=model_root_path,
                 pseudo_train_dir=target_path,
                 test_dir=test_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unlabeled_path', type=str, default='unlabeled/', help='Path of unlabeled dataset used for pseudo training')
    parser.add_argument('--target_path', type=str, default='pseudo/', help='Path to store images of unlabeled dataset along with the pseudo labels')
    parser.add_argument('--synthetic_validation_path', type=str, default='validation_synthetic/', help='Path of the synthetic validation set')
    parser.add_argument('--test_path', type=str, default='C1/', help='Path of the real test set')
    parser.add_argument('--arch', type=str, default='swin/', help='Architecture used in pseudo training')
    parser.add_argument('--model_root_path', type=str, default='models/', help='Root path of stored PL models')

    args, _ = parser.parse_known_args()

    generate_and_train(unlabeled_set=args.unlabeled_path,target_path=args.target_path,model_root_path=args.model_root_path,arch=args.arch,synthetic_validation_path=args.synthetic_validation_path,test_path=args.test_path)