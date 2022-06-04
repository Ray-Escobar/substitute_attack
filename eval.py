import torch

def load_model(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    return model

def eval_image(device, model, X, Y):
    # single image evaluation
    model.eval() # set to eval mode
    model.to(device)
    with torch.no_grad():

        X = X.to(device).unsqueeze(0)

        pred = model(X)
        # max argument from 1 dimensional array
        print('expected:', Y.item())
        print('sample prediction:', pred.argmax(1).item())


def generally_correct(device, models, x, y):
    '''
    Checks if image is correctly clasified by 
    all models.
    '''

    with torch.no_grad():
        x = x.to(device).unsqueeze(0)
        for model in models:
            model.eval()
            model.to(device)    
            _, y_pred = model(x).max(1)
            if y.item() != y_pred.item():
                return False
    return True


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
    pass

    # train_resnet_34(dataloaders)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if device == "cpu":
    #     print("cuda not working")
    # else:
    #     model = resnet18.full_resnet_18(device, pretrained=False)
    #     model = load_model(model, './trained_models/resnet_18_ox.pth')
    #     criterion = nn.CrossEntropyLoss()
    #     dataloaders = prepare_data()
    #     eval_model(model, dataloaders['test'], criterion, device)
    