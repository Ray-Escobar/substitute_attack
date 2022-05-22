from torchvision import models
from torch import nn
import torch

def full_goog_le_net(device, pretrained=True):
    '''
    For finetuning resnet18 convNet
    '''
    model = models.googlenet(pretrained=pretrained)

    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    return model

def load_goog_le_net(device, path):
    '''
    Load a goog_le_net from some file
    '''
    model = models.googlenet(pretrained=False)

    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load(path))
    model.eval()
    
    model = model.to(device)

    return model