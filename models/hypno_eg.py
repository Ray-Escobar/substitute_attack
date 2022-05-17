# def hypno():
#     # Data augmentation and normalization for training
#     # Just normalization for validation
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }

#     data_dir = './datasets/hymenoptera_data'
#     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                             data_transforms[x])
#                     for x in ['train', 'test']}
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                                 shuffle=True, num_workers=4)
#                 for x in ['train', 'test']}
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
#     class_names = image_datasets['train'].classes

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     model_ft = models.resnet18(pretrained=True)
#     num_ftrs = model_ft.fc.in_features
#     # Here the size of each output sample is set to 2.
#     # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#     model_ft.fc = nn.Linear(num_ftrs, 2)

#     model_ft = model_ft.to(device)

#     criterion = nn.CrossEntropyLoss()

#     # Observe that all parameters are being optimized
#     optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

#     # Decay LR by a factor of 0.1 every 7 epochs
#     exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#     model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=25, device=device)
            
