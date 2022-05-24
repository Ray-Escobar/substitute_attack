from torchvision import models
from torch import nn

def full_vgg16(device, pretrained=True):
    '''
    Full pretrained vgg_16 convNet
    '''
    model = models.vgg16(pretrained=pretrained)
    # model.classifier._modules['6'] = nn.Linear(4096, 2)
    model.classifier =  model.classifier.append(nn.Linear(1000, 2))
    model = model.to(device)

    return model