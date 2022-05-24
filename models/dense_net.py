from torchvision import models
from torch import nn
import torch

def full_dense(device, pretrained=True):
    '''
    For finetuning resnet18 convNet
    '''
    model = models.densenet121(pretrained=pretrained)

    model.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)

    # model.classifier._modules['2'] = nn.Linear(in_features=768, out_features=2, bias=True)
    # num_ftrs = model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    return model

def load_dense(path):
    '''
    Load a resnet_18 from some file
    '''
    model = models.densenet121(pretrained=False)

    model.name = "densenet_121"

    model.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)

    model.load_state_dict(torch.load(path))

    return model