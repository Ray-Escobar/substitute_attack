from logging import root
import time
import copy
import torch
import os
from torch import nn, optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from data import OX_STATS, STANDARD_TRANSFORM, TRAIN_TRANSFORM, OxfordPetsDataset, split_dataframe
import models.resnet18 as resnet18
import models.resnet50 as resnet50
import models.googLeNet as googLeNet
import models.vgg as vgg
import models.convNeXt as cnx
import models.dense_net as dense
from attacks import eval_loop
import json

def models_to_train():
    return {
        'goog_le_net': googLeNet.full_goog_le_net,
        'dense_121' : dense.full_dense,
        'res_net_50': resnet50.full_resnet_50,
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
        'train' : len(dataloaders['train'].dataset),
        'test'  : len(dataloaders['test'].dataset)   
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
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
        
        if best_acc >= 0.99:
            print('Accuracy greater than 0.99 achieved. Stopping early')
            break

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

def prepare_symmetric_data(img_dir='./datasets/oxford-pets/images'):

    # Apply transformations to the train dataset
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=OX_STATS["mean"], std=OX_STATS["std"])
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
        'train': DataLoader(dataset=ox_dataset_train, batch_size=64, shuffle=True, num_workers=2),
        'test' : DataLoader(dataset=ox_dataset_test, batch_size=64, shuffle=False, num_workers=2)
    }
    
    return dataloaders

def export_attributes(proj_name, attribute_name, attributes):
    attributes['mean'] = ["%.3f" % number for number in attributes['mean']]
    attributes['std'] = ["%.3f" % number for number in attributes['std']]
    j_string = json.dumps(attributes)
    # Using a JSON string
    with open(f'./trained_models/{proj_name}/{attribute_name}.json', 'w') as outfile:
        outfile.write(j_string)


def train_disjoint_models(device, proj_name, csv_path, cross_section = 0):
    if cross_section == 0:
        print('Starting Disjoint Training')
    if cross_section > 0:
        print(f'Starting Cross-Sectiom Training: {cross_section*2}')
        
    print('-'*20)
    os.mkdir(f'./trained_models/{proj_name}')
    models_map = models_to_train()
    
    target_data, subs_data, target_stats, subs_stats = split_dataframe(csv_path=csv_path, cross_section=cross_section)

    export_attributes(proj_name, 'target', target_stats)
    export_attributes(proj_name, 'substitute', subs_stats)

    target_dl = create_dataloaders(target_data)
    substitute_dl = create_dataloaders(subs_data)
    criterion = nn.CrossEntropyLoss()
    
    for name, gen_model in models_map.items():
        print("Model name", name)
        print("Training target")
        torch.cuda.empty_cache()
        target = gen_model(device)
        optimizer_conv = optim.Adam(target.parameters(), lr=0.001) #, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

        
        target_model = train_model(target, target_dl, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=9, device=device)
        persist_model(target_model, f"{name}_target", path=f'./trained_models/{proj_name}')

        print()
        torch.cuda.empty_cache()
        print("Training substitute")
        substitute = gen_model(device)
        optimizer_conv = optim.Adam(substitute.parameters(), lr=0.001) #, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

        subs_model = train_model(substitute, substitute_dl, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=9, device=device)
        persist_model(subs_model, f"{name}_subs", path=f'./trained_models/{proj_name}')


def train_symmetric_models(device, proj_name, adv=True):
    '''
    adv - if True it trains a subsitute as well
    '''
    print('Training Symmetric Models')
        
    print('-'*20)
    os.mkdir(f'./trained_models/{proj_name}')
    models_map = models_to_train()
    
    dataloaders = prepare_symmetric_data()
    criterion = nn.CrossEntropyLoss()
    
    for name, gen_model in models_map.items():
        print("Model name", name)
        print("Training target")
        torch.cuda.empty_cache()
        target = gen_model(device)
        optimizer_conv = optim.Adam(target.parameters(), lr=0.001) #, momentum=0.9)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

        
        target_model = train_model(target, dataloaders, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=9, device=device)
        persist_model(target_model, f"{name}_target", path=f'./trained_models/{proj_name}')

        if adv:
            print()
            torch.cuda.empty_cache()
            print("Training substitute")
            substitute = gen_model(device)
            optimizer_conv = optim.Adam(substitute.parameters(), lr=0.001) #, momentum=0.9)
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

            subs_model = train_model(substitute, dataloaders, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=9, device=device)
            persist_model(subs_model, f"{name}_subs", path=f'./trained_models/{proj_name}')


def create_dataloaders(datasets):
    return {
        'train': DataLoader(dataset=datasets['train'], batch_size=64, shuffle=True, num_workers=2),
        'test' : DataLoader(dataset=datasets['test'], batch_size=64, shuffle=False, num_workers=2)
    }


if __name__ == "__main__":

    # dataloaders = prepare_data()
    csv_path = './datasets/oxford-pets/annotations/list.txt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: cpu training is very slow!")

    # dense.full_dense(device)
    # torch.cuda.empty_cache()
    train_symmetric_models(device, 'adv_data_gens')
    # train_disjoint_models(device, "cross_section_2", csv_path, cross_section=8)
    # train_disjoint_models(device, "disjoint_2", csv_path, cross_section=0)
    