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

OX_STATS_2 = {
    "mean" : (0.485, 0.456, 0.406),
    "std"  : (0.229, 0.224, 0.225)
}
# https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276

class OxfordPetsDataset(Dataset):
    """
    Args:
        file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_path, img_dir, row_skips=0, transform=None):

        self.data_frame = pd.read_csv(csv_path, sep=" ", skiprows=row_skips)
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



def normalize_params(dataset):
    '''
    calculates the mean and std of each RGB channel

    dataset - img dataset to calcualte stats for
    '''

    imgs = [item[0] for item in dataset] # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()


    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    mean = (mean_r,mean_g,mean_b)

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    std = (std_r,std_g,std_b)

    return mean, std


def torch_normalize(dataset):
    '''
    Whole image normalization
    '''
    means = []
    stds = []
    for img, _ in dataset:
        means.append(torch.mean(img))
        stds.append(torch.std(img))

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))

    return mean, std

def inverse_normalize(tensor, mean=(0.48141265, 0.44937795, 0.39572072), std=(0.26479402, 0.2600657, 0.26857644)):
    '''
    inverses a normalization
    '''
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    tensor.mul_(std).add_(mean)
    return tensor

def read_oxford_pets_csv():
    df = pd.read_csv("./datasets/oxford-pets/annotations/list.txt", sep=" ", skiprows=6)
    breed_id = np.asarray(df.iloc[:, 1:3])#egyptian mau 186

    breeds = np.asarray(df.iloc[:, [0][0]])

    breed_map = {}
    for id, breed in zip(breed_id, breeds):
        breed = re.split('_[0-9]', breed)[0]
        breed_map[id[0]] = (id[1], str.lower(breed))
    for item in breed_map.items():
        if item[1][0] == 1:
            print(item)

def split_dataset(dataset, train_split=0.80):
    '''
    Splits a given dataset into train and test

    train_split - percentage of data used for training
    dataset     - whole dataset
    '''
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def plot_tensor_img(img, label):
    '''
    plots a single tensored image
    '''
    img = np.transpose(copy.deepcopy(img), (1, 2, 0))
    plt.imshow(img)
    plt.show()



if __name__ == "__main__":
    # I think normalizing the data should be enough!
    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.Random3734HorizontalFlip(),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(ox_stats["mean"], ox_stats["std"])
    ])


    ox_dataset = OxfordPetsDataset(
        csv_path = './datasets/oxford-pets/annotations/list.txt', 
        img_dir = './datasets/oxford-pets/images',
        transform = data_transform
    )
    # datasets.OxfordIIIPet(root='./data/cifar10', train=True, download=True, transform=transforms.ToTensor())

    print(len(ox_dataset))

    train, test = split_dataset(ox_dataset)

    print(len(train), len(test), len(train) + len(test))
    # img, y = ox_dataset[16]
    # img = inverse_normalize(img)
    # print(img.shape[0] == 3)
    # plot_tensor_img(img, y)
    # print(ox_dataset.__len__())
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()

    



# def read_cats_csv():
    
#     classes = {}
#     cat_dataset = datasets.ImageFolder(root='./datasets/catz/images', transform=data_transform)
#     cats_frame = pd.read_csv('datasets/catz/data/cats.csv')

#     b_n = 8
#     breeds = np.asarray(cats_frame.iloc[:, [8]])

#     unique_breeds = np.unique(breeds)

#     print(unique_breeds)
#     print("unique breeds:", len(unique_breeds))
