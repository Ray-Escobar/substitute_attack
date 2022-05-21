from logging import root
import time
import copy
import torch
import os
from torch import nn, optim
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from data import OX_STATS_2, OxfordPetsDataset, split_dataset, OX_STATS, SPECIES


def models_to_train():
    return {
        'res_net_18': ()
    }

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    '''
    train_model is a general function to train any given model.

    model - model to train
    criterion - loss function
    optimizer - (SGD or Adams)
    scheduler - decreases learning rate as epochs increase
    num_epochs - amount of times to run learning step
    '''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {
        'train' : len(dataloaders['train']),
        'test'  : len(dataloaders['test'])   
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
                    # only in backwards we actually propagate the gradients!
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

def eval_model(model, dataloader, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


def persist_model(model, name, path='./models'):
    # Persist the Model
    path = os.path.join(path, f'{name}.pth')
    torch.save(model.state_dict(), path)

def train_resnet_50(dataloaders):
    # Freeze Training
    model = models.resnet50(pretrained = True)
    
    num_ftrs = model.fc.in_features


    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to("cpu")

    # for param in model_freeze.parameters():
    #     param.requires_grad = False

    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model_freeze.fc.in_features
    # model_freeze.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()


    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.Adam(model.parameters(), lr=0.0001) #, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model = train_model(model, dataloaders, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=10)

    persist_model(model, "resnet_34_ox1")


if __name__ == "__main__":




    # Apply transformations to the train dataset
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # apply the same transformations to the validation set, with the exception of the
    # randomized transformation. We want the validation set to be consistent
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ox_dataset_train = OxfordPetsDataset(
        csv_path = './datasets/oxford-pets/annotations/test.txt', 
        img_dir = './datasets/oxford-pets/images',
        transform = train_transforms
    )

    ox_dataset_test = OxfordPetsDataset(
        csv_path = './datasets/oxford-pets/annotations/trainval.txt', 
        img_dir = './datasets/oxford-pets/images',
        transform = test_transforms
    )
    # train_data, test_data = split_dataset(ox_dataset)

    dataloaders = {
        'train': DataLoader(dataset=ox_dataset_train, batch_size=64, shuffle=True, num_workers=2),
        'test' : DataLoader(dataset=ox_dataset_test, batch_size=64, shuffle=False, num_workers=2)
    }

    train_resnet_34(dataloaders)

