import os
import re
import copy
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torchvision import transforms


OX_FILE_COLS = ["Image", "Id", "Species", "Breed_Id"]

SPECIES = {
    0 : "cat",
    1 : "dog"
}

# breed id and breed name
OX_BREEDS = {
    1: 'abyssinian', 
    2: 'american_bulldog', 
    3: 'american_pit_bull_terrier', 
    4: 'basset_hound', 
    5: 'beagle', 
    6: 'bengal', 
    7: 'birman', 
    8: 'bombay', 
    9: 'boxer', 
    10: 'british_shorthair', 
    11: 'chihuahua', 
    12: 'egyptian_mau', 
    13: 'english_cocker_spaniel', 
    14: 'english_setter', 
    15: 'german_shorthaired', 
    16: 'great_pyrenees', 
    17: 'havanese', 
    18: 'japanese_chin', 
    19: 'keeshond', 
    20: 'leonberger', 
    21: 'maine_coon', 
    22: 'miniature_pinscher', 
    23: 'newfoundland', 
    24: 'persian', 
    25: 'pomeranian', 
    26: 'pug', 
    27: 'ragdoll', 
    28: 'russian_blue', 
    29: 'saint_bernard', 
    30: 'samoyed', 
    31: 'scottish_terrier', 
    32: 'shiba_inu', 
    33: 'siamese', 
    34: 'sphynx', 
    35: 'staffordshire_bull_terrier', 
    36: 'wheaten_terrier', 
    37: 'yorkshire_terrier'
    }

BREEDS_ID = {
    "dogs" : [2, 3, 4, 5, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 29, 30, 31, 32, 35, 36, 37],
    "cats" : [1, 6, 7, 8, 10, 12, 21, 24, 27, 28, 33, 34]
}

OX_STATS = {
    "mean" : (0.48141265, 0.44937795, 0.39572072),
    "std"  : (0.26479402, 0.2600657, 0.26857644)
} 


STANDARD_TRANSFORM = lambda : transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

TRAIN_TRANSFORM = lambda : transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

# https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276
class OxfordPetsDataset(Dataset):
    """
    Args:
        file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_path, img_dir, row_skips=0, transform=None):

        self.data_frame = pd.read_csv(csv_path, sep=" ", skiprows=row_skips) # manipualte dataframe!!!
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

class OxfordPetsDatasetSplit(Dataset):
    """
    Args:
        file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        split_dog (list) : lists containg breed ids and only picks those from the dataframe
    """

    def __init__(self, csv_path, img_dir, split = [], row_skips=0, transform=None):

        frame = pd.read_csv(csv_path, sep=" ", skiprows=row_skips, names=OX_FILE_COLS)
        self.data_frame = frame.iloc[split]
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

def get_class_indices(dataset, class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices

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
    mean = [mean_r,mean_g,mean_b]

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    std = [std_r,std_g,std_b]

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
    breed_id = np.asarray(df.iloc[:, 1:3])

    breeds = np.asarray(df.iloc[:, [0][0]])

    breed_map = {}
    for id, breed in zip(breed_id, breeds):
        breed = re.split('_[0-9]', breed)[0]
        breed_map[id[0]] = str.lower(breed) #(id[1], str.lower(breed))
    
    dogs = []
    cats = []
    for item in breed_map.items():
        # print(item)
        if item[1][0] == 1:
            cats.append(item[0])
        else:
            dogs.append(item[0])
    return cats, dogs

def __cats_dogs_split(drop_last=True):
    '''
    Splits thhe breeds of cats and dogs each 
    '''

    c1, c2 = __split_list(BREEDS_ID["cats"])
    d1, d2 =__split_list(BREEDS_ID["dogs"])
    if drop_last:
        return np.concatenate((c1, d1)), np.concatenate((c2, d2[:-1]))
    else:
        return np.concatenate((c1, d1)), np.concatenate((c2, d2))        

def __split_list(items):
    '''
    Returns 
    '''
    picks = np.random.choice(items, len(items)//2, replace=False)
    rest = np.setdiff1d(items, picks)
    return picks, rest

def log_split(labels1, labels2):
    print('| Target Split |')
    for label in labels1:
        print(OX_BREEDS[label])

    print()
    print('| Substitute Split |')
    for label in labels2:
        print(OX_BREEDS[label])
    print('-' * 20)
    print()

def log_stats(label, stats):
    print(f'| {label} stats |')
    print(f'Mean: {stats[0]}')
    print(f'Std: {stats[1]}')
    print('-' * 20)
    print()


def split_dataframe(csv_path='./datasets/oxford-pets/annotations/list.txt', img_dir='./datasets/oxford-pets/images', test_size=0.2, skip_rows = 6):
    """
    Spits the dataset given a csv file and returns the
    split datasets into the black-box model and the substitute model.

    returens:
    - maps of target and substitute train and test data
    - 
    """
    # split label breeds disjointly
    labels1, labels2 = __cats_dogs_split(drop_last=True)
    
    frame = pd.read_csv(csv_path, sep=" ", skiprows=skip_rows, names=OX_FILE_COLS)
    idxs_target, idxs_sub = frame.index[frame.Id.isin(labels1)].to_list(), frame.index[frame.Id.isin(labels2)].to_list()

    stats_target = normalize_params(OxfordPetsDatasetSplit(
            csv_path = csv_path, 
            img_dir = img_dir,
            split = idxs_target,
            row_skips = skip_rows,
            transform = STANDARD_TRANSFORM()
            ))

    stats_sub = normalize_params(OxfordPetsDatasetSplit(
            csv_path = csv_path, 
            img_dir = img_dir,
            split = idxs_sub,
            row_skips = skip_rows,
            transform= STANDARD_TRANSFORM()
            ))

    log_split(labels1, labels2)
    log_stats('Target', stats_target)
    log_stats('Subs', stats_sub)
    target_data = __create_model_dataset(csv_path, idxs_target, stats_target, test_size)
    subs_data = __create_model_dataset(csv_path,  idxs_sub, stats_sub, test_size)
    return target_data, subs_data, {
        "split": labels1.tolist(), 
        "mean": stats_target[0],
        "std": stats_target[1]
        }, {"split" : labels2.tolist(), 
            "mean"  :  stats_sub[0],
            "std"   :  stats_sub[1]
            } 


def __create_model_dataset(csv_path, idxs, stats, test_size=0.2):
    t_train = TRAIN_TRANSFORM()
    t_test  = STANDARD_TRANSFORM()
    t_train.transforms.append(transforms.Normalize(stats[0], stats[1]))
    t_test.transforms.append(transforms.Normalize(stats[0], stats[1]))
    train_idxs, test_idxs = train_test_split(idxs, test_size=test_size, random_state=25)
    model_dataset = {
        "train": OxfordPetsDatasetSplit(
            csv_path = csv_path, 
            img_dir = './datasets/oxford-pets/images',
            split = train_idxs,
            row_skips=6,
            transform = t_train
        ),
        "test": OxfordPetsDatasetSplit(
            csv_path = csv_path, 
            img_dir = './datasets/oxford-pets/images',
            split = test_idxs,
            row_skips=6,
            transform = t_test
        )
    }

    # print(len(black_box_data["train"]), len(black_box_data["test"]))
    # print(black_box_data["train"].transform, black_box_data["test"].transform)
    return model_dataset

if __name__ == "__main__":

    csv_path = './datasets/oxford-pets/annotations/list.txt'
    split_dataframe(csv_path)




    # img, y = ox_dataset[16]
    # img = inverse_normalize(img)
    # print(img.shape[0] == 3)
    # plot_tensor_img(img, y)
    # print(ox_dataset.__len__())
    # print(img.shape)
    # # img = inverse_normalize(img)
    # # print(img.shape[0] == 3)
    # # plot_tensor_img(img, y)
    # # print(ox_dataset.__len__())
    # # print(img.shape)
    # plt.imshow(img)
    # plt.show()