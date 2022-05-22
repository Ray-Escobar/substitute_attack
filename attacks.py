import torch
from torch import nn
from data import OxfordPetsDataset, STANDARD_TRANSFORM
from torch.utils.data import DataLoader
import numpy as np
import models.resnet18 as resnet18
import models.resnet50 as resnet50
import models.googLeNet as googLeNet
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

def eval_loop(dataloader, model, criterion):
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
        
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
    
    return report

def transfer_attack(device, target, substitute, data, eps):
    '''
    target - some model
    substitute - model from which adversarial images will be generated from
    '''
    target.eval()
    substitute.eval()

    target.to(device)
    substitute.to(device)

    # loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    # loss_fn = nn.CrossEntropyLoss()

    report = {'nb_test' : 0, 'correct': 0, 'correct_fgm':0, 'correct_pgd': 0}
    for x, y in data:
        x, y = x.to(device), y.to(device)
        
        # model prediction on clean examples
        _, y_pred = target(x).max(1)

        x_fgm = fast_gradient_method(substitute, x, eps, np.inf) #clip_min= -4, clip_max=4)
        x_pgd = projected_gradient_descent(substitute, x, eps, 0.01, 40, np.inf)# clip_min= -4, clip_max=4)

        # model prediction on FGM adversarial examples
        _, y_pred_fgm = target(x_fgm).max(1)  
        
        # model prediction on PGD adversarial examples
        _, y_pred_pgd = target(x_pgd).max(1)  

        # examples.append((x_fgm.squeeze().detach().cpu().numpy(), x_pgd.squeeze().detach().cpu().numpy(), y_pred_fgm, y_pred_pgd, y_pred))
        
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
    
    return report


def preform_attack_cross_section(device, targets, substitutes):
    '''
    targets - list of targets models
    substitutes - list of substitute models
    '''
    for substitute in substitutes:
        for target in targets:
            torch.cuda.empty_cache()
            transfer_attack(device, target, substitute)


def print_report(report):
    '''
    Given a report, it prints it out
    '''
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    print("test acc on FGM adversarial examples (%): {:.3f}".format(
            report.correct_fgm / report.nb_test * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report.correct_pgd / report.nb_test * 100.0
        )
    )

if __name__ == "__main__":
    models_dir = './trained_models'
    img_dir='./datasets/oxford-pets/images'
    csv_path = './datasets/oxford-pets/annotations/trainval.txt'
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: cpu evaluation is very slow!")

    targets = [
        resnet18.load_resnet_18(device, f'{models_dir}/resnet_18_ox.pth'),
        googLeNet.load_goog_le_net(device, f'{models_dir}/goog_le_net.pth')
    ]

    substitutes = [
        resnet18.load_resnet_18(device, f'{models_dir}/resnet_18_ox_adv.pth'),
        googLeNet.load_goog_le_net(device, f'{models_dir}/goog_le_net_adv.pth')
    ]

    criterion = nn.CrossEntropyLoss()

    ox_dataset_test = OxfordPetsDataset(
        csv_path = csv_path, 
        img_dir = img_dir,
        transform = STANDARD_TRANSFORM
    )

    dataloader = DataLoader(dataset=ox_dataset_test, batch_size=20, shuffle=True, num_workers=2)

    # eval_loop(dataloader, model, criterion)
    for m in targets:
        eval_loop(dataloader, m, criterion)
