import copy
import numpy as np
from data import OX_STATS
import matplotlib.pyplot as plt

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
