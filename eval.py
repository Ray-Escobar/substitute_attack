import torch
import models.resnet18 as resnet18
from torch import nn
from train import prepare_data

def load_model(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    return model

def eval_model(model, dataloader, criterion, device):
    model.eval() # set to eval mode

    model.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    print('Evaluating Model')
    print('-' * 10)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs)
            test_loss += criterion(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":

    # train_resnet_34(dataloaders)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("cuda not working")
    else:
        model = resnet18.full_resnet_18(device, pretrained=False)
        model = load_model(model, './trained_models/resnet_18_ox.pth')
        criterion = nn.CrossEntropyLoss()
        dataloaders = prepare_data()
        eval_model(model, dataloaders['test'], criterion, device)
    