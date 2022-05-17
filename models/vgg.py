from torchvision import models
from torch import nn

def full_vgg_19(device, pretrained=True):
    '''
    For finetuning resnet18 convNet
    '''
    model = models.vgg19(pretrained=pretrained)

    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    return model