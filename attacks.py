import torch
from torch import nn
from data import OxfordPetsDataset, STANDARD_TRANSFORM, OX_STATS
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import models.resnet50 as resnet50
import models.googLeNet as googLeNet
import models.dense_net as dense
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

def eval_loop(device, dataloader, model, criterion):
    print("Evaluating model:", model.name)
    model.eval()
    model.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


def eval_model_baselines(device, targets, substitutes, criterion):

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=OX_STATS["mean"], std=OX_STATS["std"])
    ])
    
    ox_dataset_test = OxfordPetsDataset(
        csv_path = './datasets/oxford-pets/annotations/trainval.txt', 
        img_dir = './datasets/oxford-pets/images',
        transform = test_transforms
    )

    dataloader = DataLoader(ox_dataset_test, batch_size=64, shuffle=False, num_workers=2)
    
    # original eval for targets
    print('Evaluating Targets')
    for m in targets:
        eval_loop(device, dataloader, m, criterion)
        torch.cuda.empty_cache()

    print()
    print('Evaluating Substitutes')
    # original eval for subs
    for m in substitutes:
        eval_loop(device, dataloader, m, criterion)
        torch.cuda.empty_cache()

def eval_attack(model, device, data, eps):
    model.eval()

    # loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    # loss_fn = nn.CrossEntropyLoss()

    report = {'nb_test' : 0, 'correct': 0, 'correct_fgm':0, 'correct_pgd': 0}
    for x, y in data:
        x, y = x.to(device), y.to(device)
        
        # model prediction on clean examples
        _, y_pred = model(x).max(1)

        x_fgm = fast_gradient_method(model, x, eps, np.inf) #clip_min= -4, clip_max=4)
        x_pgd = projected_gradient_descent(model, x, eps, 0.01, 40, np.inf)# clip_min= -4, clip_max=4)

        # model prediction on FGM adversarial examples
        _, y_pred_fgm = model(x_fgm).max(1)  
        
        # model prediction on PGD adversarial examples
        _, y_pred_pgd = model(x_pgd).max(1)  

        # examples.append((x_fgm.squeeze().detach().cpu().numpy(), x_pgd.squeeze().detach().cpu().numpy(), y_pred_fgm, y_pred_pgd, y_pred))
        
        report['nb_test'] += y.size(0)
        report['correct'] += y_pred.eq(y).sum().item()
        report['correct_fgm'] += y_pred_fgm.eq(y).sum().item()
        report['correct_pgd'] += y_pred_pgd.eq(y).sum().item()
    
    return report

def transfer_attack(device, target, substitute, dataloader, eps):
    '''
    target - some model
    substitute - model from which adversarial images will be generated from
    '''

    print('Target:', target.name, 'Substitute:', substitute.name)
    target.eval()
    substitute.eval()

    target.to(device)
    substitute.to(device)

    # loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    # loss_fn = nn.CrossEntropyLoss()

    report = {'nb_test' : 0, 'correct': 0, 'correct_fgm':0, 'correct_pgd': 0}
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        # model prediction on clean examples
        _, y_pred = target(x).max(1)

        # generate adversary with substitute!
        x_fgm = fast_gradient_method(substitute, x, eps, np.inf) #clip_min= -4, clip_max=4)
        x_pgd = projected_gradient_descent(substitute, x, eps, 0.01, 40, np.inf)# clip_min= -4, clip_max=4)

        # model prediction on FGM adversarial examples
        _, y_pred_fgm = target(x_fgm).max(1)  
        
        # model prediction on PGD adversarial examples
        _, y_pred_pgd = target(x_pgd).max(1)  

        # examples.append((x_fgm.squeeze().detach().cpu().numpy(), x_pgd.squeeze().detach().cpu().numpy(), y_pred_fgm, y_pred_pgd, y_pred))
        
        report['nb_test'] += y.size(0)
        report['correct'] += y_pred.eq(y).sum().item()
        report['correct_fgm'] += y_pred_fgm.eq(y).sum().item()
        report['correct_pgd'] += y_pred_pgd.eq(y).sum().item()
    
    return report


def perform_attack_cross_section(device, targets, substitutes, dataloader, eps):
    '''
    targets - list of targets models
    substitutes - list of substitute models
    '''
    torch.cuda.empty_cache()
    for substitute in substitutes:
        for target in targets:
            torch.cuda.empty_cache()
            report = transfer_attack(device, target, substitute, dataloader, eps)
            print_report(report)

    torch.cuda.empty_cache()

def print_report(report):
    '''
    Given a report, it prints it out
    '''
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report['correct'] / report['nb_test'] * 100.0
        )
    )
    print("test acc on FGM adversarial examples (%): {:.3f}".format(
            report['correct_fgm'] / report['nb_test'] * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report['correct_pgd'] / report['nb_test'] * 100.0
        )
    )

def symmetric_attack(device, dataloader, models_dir='./trained_models/symmetric', eval_baseline= False):
    print('Running Symmetric Attack')
    print('-'*20)
    criterion = nn.CrossEntropyLoss()
    targets = [
        resnet50.load_resnet_50(f'{models_dir}/res_net_50_target.pth'),
        googLeNet.load_goog_le_net(f'{models_dir}/goog_le_net_target.pth'),
        dense.load_dense(f'{models_dir}/dense_121_target.pth')
    ]

    substitutes = [
        resnet50.load_resnet_50(f'{models_dir}/res_net_50_target.pth'),
        googLeNet.load_goog_le_net(f'{models_dir}/goog_le_net_subs.pth'),
        dense.load_dense(f'{models_dir}/dense_121_subs.pth')
    ]

    if eval_baseline:
        eval_model_baselines(device, targets, substitutes, criterion)

    perform_attack_cross_section(device, targets, substitutes, dataloader, 0.1)


def disjoint_attack(device, dataloader, models_dir='./trained_models/first_run_disjoint', eval_baseline= False):
    print('Running Disjoint Attack')
    print('-'*20)
    criterion = nn.CrossEntropyLoss()
    targets = [
        googLeNet.load_goog_le_net(f'{models_dir}/goog_le_net_target.pth'),
        resnet50.load_resnet_50(f'{models_dir}/res_net_50_target.pth'),
        dense.load_dense(f'{models_dir}/dense_121_target.pth')
    ]

    substitutes = [
        googLeNet.load_goog_le_net(f'{models_dir}/goog_le_net_subs.pth'),
        resnet50.load_resnet_50(f'{models_dir}/res_net_50_subs.pth'),
        dense.load_dense(f'{models_dir}/dense_121_subs.pth')
    ]

    if eval_baseline:
        eval_model_baselines(device, targets, substitutes, criterion)

    perform_attack_cross_section(device, targets, substitutes, dataloader, 0.1)


if __name__ == "__main__":
    print("| Starting attacks |")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: cpu evaluation is very slow!")

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=OX_STATS["mean"], std=OX_STATS["std"])
    ])
    
    ox_dataset_test = OxfordPetsDataset(
        csv_path = './datasets/oxford-pets/annotations/list.txt', 
        img_dir = './datasets/oxford-pets/images',
        row_skips= 6,
        transform = test_transforms
    )
    
    samples = np.random.choice(len(ox_dataset_test), 50, replace=False)

    attack_set = torch.utils.data.Subset(ox_dataset_test, samples)
    
    dataloader = DataLoader(attack_set, batch_size=4, shuffle=False, num_workers=2)

    # symmetric_attack(device, dataloader)
    symmetric_attack(device, dataloader, eval_baseline=False, models_dir='./trained_models/symmetric_1')
    disjoint_attack(device, dataloader, eval_baseline=False, models_dir='./trained_models/cross_section_2/')
    disjoint_attack(device, dataloader, eval_baseline=False, models_dir='./trained_models/disjoint_2')