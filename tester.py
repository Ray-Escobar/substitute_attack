import os
import re
import copy
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


SPECIES = {
    0 : "cat",
    1 : "dog"
}

OX_STATS = {
    "mean" : (0.48141265, 0.44937795, 0.39572072),
    "std"  : (0.26479402, 0.2600657, 0.26857644)
} 

class OxfordPetsDataset(Dataset):
    """
    Args:
        file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_path, img_dir, transform=None):

        self.data_frame = pd.read_csv(csv_path, sep=" ", skiprows=6)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        # This method should return only 1 sample and label 
        # (according to "index"), not the whole dataset
        # So probably something like this for you:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, f'{self.data_frame.iloc[idx, 0]}.jpg')
        sample  = io.imread(img_name) #, self.data_frame.iloc[idx, 2]
        species = self.data_frame.iloc[idx, 2]

        if self.transform:
            sample = self.transform(sample)

        # subs 1 to labels to bring it down to 0, 1
        return sample, torch.tensor(species-1)

    def __len__(self):
        return len(self.data_frame)



if __name__ == "__main__":

    print("wow test")
    data_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(OX_STATS["mean"], OX_STATS["std"])
        ])

    ox_dataset = OxfordPetsDataset(
        csv_path = './datasets/oxford-pets/annotations/list.txt', 
        img_dir = './datasets/oxford-pets/images',
        transform = data_transform
    )
