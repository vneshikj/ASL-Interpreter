import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset

# https://www.kaggle.com/code/vijaypro/cnn-pytorch-96#Training-the-model

class ConvertDataset(Dataset):

    def __init__(self, csv, train=True):
        self.classes = list(range(26))
        self.csv = pd.read_csv(csv)
        self.imgSize = (28, 28, 1)
        self.train = train
        text = "pixel"
        self.images = torch.zeros((self.csv.shape[0], 1))

        for i in range(1, 785):
            tempText = text + str(i)
            temp = self.csv[tempText]
            temp = torch.FloatTensor(temp).unsqueeze(1)
            self.images = torch.cat((self.images, temp), 1)

        self.labels = self.csv['label']
        self.images = self.images[:, 1:]
        self.images = self.images.view(-1, 28, 28)

    def __getitem__(self, index):
        img = self.images[index]
        img = img.numpy()
        img = cv2.resize(img, self.imgSize[:2])
        tensorImage = torch.FloatTensor(img)
        tensorImage /= 255.
        tensorImage = tensorImage.unsqueeze(0)
        if self.train:
            return tensorImage, self.labels[index]
        else:
            return tensorImage

    def __len__(self):
        return self.images.shape[0]
