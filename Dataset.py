import cv2 as cv
import numpy as np
import torch
import os
import random
import albumentations as A
import torchvision
# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset='../Data/Dataset', setting='train', sim=True, original=False):
        self.path = dataset
        self.classes = os.listdir(self.path)
        self.interferograms = []
        self.interferograms_normal = []
        self.interferograms_deformation = []
        self.sim = sim
        self.original = original
        self.oversampling = True
        for data_class in self.classes:
            images = os.listdir(self.path + '/' + data_class)
            for image in images:
                if 'ipynb' in image:
                    continue
                image_dict = {'path': self.path + '/' + data_class + '/' + image, 'label': data_class}
                self.interferograms.append(image_dict)
                if int(data_class) == 0:
                    self.interferograms_normal.append(image_dict)
                else:
                    self.interferograms_deformation.append(image_dict)

        self.num_examples = len(self.interferograms)
        self.set = setting


    def __len__(self):
        return self.num_examples


    def __getitem__(self, index):

        if self.set == 'train' and self.sim == False and self.oversampling:
            #print('Oversampling')
            choice = random.randint(0, 10)
            buffer = False
            if choice % 2 != 0:
                choice_normal = random.randint(0, len(self.interferograms_normal) - 1)
                image_data = self.interferograms_normal[choice_normal]
            else:
                choice_deform = random.randint(0, len(self.interferograms_deformation) - 1)
                image_data = self.interferograms_deformation[choice_deform]
        else:
            image_data = self.interferograms[index]

        image_file = image_data['path']
        image_label = image_data['label']
        image = cv.imread(image_file)
        zero = np.zeros_like(image)
        if image is None:
            print(image_file)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        original = image
        original = original[:224, :224, :]
        zero[:, :, 0] = gray
        zero[:, :, 1] = gray
        zero[:, :, 2] = gray
        image = zero
        image = image[:224, :224, :]

        image = torch.from_numpy(image).float().permute(2, 0, 1)
        original = torch.from_numpy(original).float().permute(2, 0, 1)

        image = torchvision.transforms.Normalize((108.6684,108.6684, 108.6684), (109.1284, 109.1284, 109.1284))(image)

        if image.shape[1] < 224 or image.shape[2] < 224:
            print(image_file)
        if self.original:
            return (image, image, original), int(image_label), image_file
        return (image, original), int(image_label)