from logging import root
import time
import copy
import torch
import os
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from data import OX_STATS, OX_STATS_2, OxfordPetsDataset
import models.resnet18 as resnet18
import models.resnet50 as resnet50
import models.googLeNet as googLeNet
import models.vgg as vgg

'''
train_model is a general function to train any given model.

model - model to train
criterion - loss function
optimizer - (SGD or Adams)
scheduler - decreases learning rate as epochs increase
num_epochs - amount of times to run learning step
'''
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {
        'train' : len(dataloaders['train'].dataset),
        'test'  : len(dataloaders['test'].dataset)   
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # scheduler decrease learning rate as epochs increase
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f}')

    # load best model weights
    # return the new trained model!
    model.load_state_dict(best_model_wts)
    return model

def persist_model(model, name, path='./trained_models'):
    # Persist the Model
    path = os.path.join(path, f'{name}.pth')
    torch.save(model.state_dict(), path)

def prepare_data(img_dir='./datasets/oxford-pets/images'):

    # Apply transformations to the train dataset
    # normal_transforms = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=OX_STATS_2["mean"], std=OX_STATS_2["std"])
    # ])

    # ox_dataset = OxfordPetsDataset(
    #     csv_path = './datasets/oxford-pets/annotations/list.txt', 
    #     img_dir = img_dir,
    #     transform = normal_transforms,
    #     row_skips=6
    # )

    # # print(type(ox_dataset))
    # ox_dataset_train, ox_dataset_test = split_dataset(ox_dataset)

    # Apply transformations to the train dataset
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=OX_STATS_2["mean"], std=OX_STATS_2["std"])
    ])

    # apply the same transformations to the validation set, with the exception of the
    # randomized transformation. We want the validation set to be consistent
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=OX_STATS["mean"], std=OX_STATS["std"])
    ])

   
    ox_dataset_train = OxfordPetsDataset(
        csv_path = './datasets/oxford-pets/annotations/test.txt', 
        img_dir = img_dir,
        transform = train_transforms
    )

    ox_dataset_test = OxfordPetsDataset(
        csv_path = './datasets/oxford-pets/annotations/trainval.txt', 
        img_dir = img_dir,
        transform = test_transforms
    )


    dataloaders = {
        'train': DataLoader(dataset=ox_dataset_train, batch_size=20, shuffle=True, num_workers=2),
        'test' : DataLoader(dataset=ox_dataset_test, batch_size=20, shuffle=False, num_workers=2)
    }
    
    return dataloaders

if __name__ == "__main__":
    dataloaders = prepare_data()

    # train_resnet_34(dataloaders)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("cuda not working")
    else:
        torch.cuda.empty_cache()
        model = resnet50.full_resnet_50(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_conv = optim.Adam(model.parameters(), lr=0.001) #, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
        model = train_model(model, dataloaders, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=10, device=device)
        persist_model(model, "resnet_50_adv")
