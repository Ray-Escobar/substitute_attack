import copy
import numpy as np
from data import OX_STATS
import matplotlib.pyplot as plt
import json
from data import OX_BREEDS

def gallery_show(inp, title):
    inp = inp.numpy().transpose((1,2,0))
    inp = OX_STATS['std'] * inp +  OX_STATS['mean'] 
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()

def plot_tensor_img(img, label):
    '''
    plots a single tensored image
    '''
    img = np.transpose(copy.deepcopy(img), (1, 2, 0))
    plt.imshow(img)
    plt.show()

def read_breed_json(path):
    # Opening JSON file
    f = open(path)
    
    # returns JSON object as 
    # a dictionary
    return json.load(f)['split']
    
    

if __name__ == "__main__":
    target_breeds = read_breed_json('./trained_models/disjoint_2/target.json')
    subs_breeds = read_breed_json('./trained_models/disjoint_2/substitute.json')

    # cross_section = np.intersect1d(target_breeds, subs_breeds)


    print('Breeds Target')
    for i, breed in enumerate(target_breeds):
        print(f'{i+1}. {OX_BREEDS[breed]}')

    print('Breeds Substitute')
    for i, breed in enumerate(subs_breeds):
        print(f'{i+1}. {OX_BREEDS[breed]}')