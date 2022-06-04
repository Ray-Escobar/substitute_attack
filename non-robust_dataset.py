import torch
import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from train import train_symmetric_models
from data import STANDARD_TRANSFORM, OX_STATS
from analysis import inverse_normalize, save_tensor_image
from eval import generally_correct
from attacks import generally_adversarial, single_attack
from torchvision import transforms
from data import OX_FILE_COLS
import pandas as pd
from torch.utils.data import DataLoader
from models import googLeNet, resnet50, dense_net



class OxfordPetsDatasetSampler(Dataset):
    """
    Args:
        file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        split_dog (list) : lists containg breed ids and only picks those from the dataframe
    """

    def __init__(self, csv_path, img_dir, num_samples, row_skips=0, transform=None):

        frame = pd.read_csv(csv_path, sep=" ", skiprows=row_skips, names=OX_FILE_COLS)
        self.data_frame = frame.sample(n=num_samples, replace=False)
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


def gaussian_noise():
    noise = torch.randn(300,300,3) * 255
    return noise.type(dtype=torch.uint8)



def generate_non_robust_data(device, models, dataset):
    print("Creating non-robust image:")
    for i, (x, y) in enumerate(dataset):
        if not generally_correct(device, models, x, y):
            continue

        torch.cuda.empty_cache()
        # run pgd on each model until its adversarial in all
        for model in models:
            adv_sample, _ , _ = single_attack(device, model, x)
            torch.cuda.empty_cache()
            if (generally_adversarial(device, models, adv_sample, y)):
                save_tensor_image(inverse_normalize(adv_sample.cpu()), f'image{i}.png')
            torch.cuda.empty_cache()

if __name__ == "__main__":
    csv_path='./datasets/oxford-pets/annotations/list.txt'
    img_dir='./datasets/oxford-pets/images'

    transform = STANDARD_TRANSFORM()
    transform.transforms.append(transforms.Normalize(OX_STATS['mean'], OX_STATS['std']))
    ox_dataset = OxfordPetsDatasetSampler(
        csv_path = csv_path, 
        img_dir = img_dir,
        row_skips=6,
        num_samples=7200,
        transform = transform
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: cpu is very slow!")

    train_symmetric_models(device, 'adv_data_gens')
    torch.cuda.empty_cache()


    goo = googLeNet.load_goog_le_net(path = './trained_models/adv_data_gens/goog_le_net_target.pth')
    res = resnet50.load_resnet_50(f'./trained_models/adv_data_gens/res_net_50_target.pth')
    dense = dense_net.load_dense(f'./trained_models/adv_data_gens/dense_121_target.pth')
    
    # plot_tensor_img(inverse_normalize(ox_dataset[0][0]), '')
    # print(generally_adversarial(device, [goo, res], ox_dataset[0][0], ox_dataset[0][1]))

    generate_non_robust_data(device, [goo, res, dense], ox_dataset)






'''
- add images of divisions
- add line graphs of accuracy worsening
'''