from torchvision import models
from torch import nn
import torch

def full_convNext(device, pretrained=True):
    '''
    For finetuning resnet18 convNet
    '''
    model = models.convnext_tiny(pretrained=pretrained)

    model.classifier._modules['2'] = nn.Linear(in_features=768, out_features=2, bias=True)
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    return model

def load_convNext(path):
    '''
    Load a goog_le_net from some file
    '''
    model = models.googlenet(pretrained=False, init_weights=True)

    model.name = 'googLeNet'
    
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load(path), strict=False)

    return model