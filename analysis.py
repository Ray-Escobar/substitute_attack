import copy
import numpy as np
from data import OX_STATS
from skimage import io
import matplotlib.pyplot as plt
import json
from data import OX_BREEDS
import torch
from PIL import Image
import torchvision.transforms as T


targets = ['goog', 'res', 'dense']

symmetric_fgsm = {
    'goog'  : [66, 86, 90],
    'res'   : [80, 4, 80],
    'dense' : [69, 70, 57]
}


cross_fgsm = {
    'goog'  : [77, 82, 90],
    'res'   : [85, 3, 71],
    'dense' : [81, 61, 59]
}

disjoint_fgsm = {
    'goog'  : [77, 86, 89],
    'res'   : [76, 6, 67],
    'dense' : [83, 75, 74]
}



symmetric_pgd = {
    'goog'  : [17, 78, 85],
    'res'   : [66, 2, 36],
    'dense' : [43, 15, 1]
}

cross_pgd = {
    'goog'  : [44, 82, 88],
    'res'   : [75, 2, 27],
    'dense' : [70, 9, 5]
}

disjoint_pgd = {
    'goog'  : [60, 85, 88],
    'res'   : [65, 5, 40],
    'dense' : [77, 44, 51]
}

def plot_accuracies():
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    students = [23,17,35,29,12]
    ax.bar(langs,students)
    plt.show()



def gallery_show(inp, title):
    inp = inp.numpy().transpose((1,2,0))
    inp = OX_STATS['std'] * inp +  OX_STATS['mean'] 
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()

    plt.show()

def read_breed_json(path):
    # Opening JSON file
    f = open(path)
    
    # returns JSON object as 
    # a dictionary
    return json.load(f)['split']
    
def inverse_normalize(tensor, mean=(0.48141265, 0.44937795, 0.39572072), std=(0.26479402, 0.2600657, 0.26857644)):
    '''
    inverses a normalization
    '''
    tensor = copy.deepcopy(tensor)
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    tensor.mul_(std).add_(mean)
    return tensor

def plot_tensor_img(tensor, label):
    '''
    plots a single tensored image
    '''
    img = np.transpose(copy.deepcopy(tensor), (1, 2, 0))
    plt.imshow(img, interpolation='nearest')
    plt.show()

def save_tensor_image(tensor, img_name, dir='.', format='PNG'):
    img = np.transpose(copy.deepcopy(tensor), (1, 2, 0))
    fig = plt.imshow(img,  interpolation='nearest')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(img_name, bbox_inches='tight', pad_inches = 0)
 


if __name__ == "__main__":
    # target_breeds = read_breed_json('./trained_models/disjoint_2/target.json')
    # subs_breeds = read_breed_json('./trained_models/disjoint_2/substitute.json')

    # # cross_section = np.intersect1d(target_breeds, subs_breeds)


    # print('Breeds Target')
    # for i, breed in enumerate(target_breeds):
    #     print(f'{i+1}. {OX_BREEDS[breed]}')

    # print('Breeds Substitute')
    # for i, breed in enumerate(subs_breeds):
    #     print(f'{i+1}. {OX_BREEDS[breed]}')
    plot_accuracies()