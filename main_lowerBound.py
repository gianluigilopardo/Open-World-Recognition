from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn

import ResNet
from dataset import *
import params
import icarl
#jfejfjefhj
#ciao
# dataset
cifar = datasets.cifar.CIFAR100
train_dataset = Dataset(dataset=cifar, train=True)
test_dataset = Dataset(dataset=cifar, train=False)

# splits
train_splits = train_dataset.splits
test_splits = test_dataset.splits

# transformers
train_transformer = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])

# data loaders
# train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE,  # num_workers=params.NUM_WORKERS,
#                          shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=params.BATCH_SIZE,  # num_workers=params.NUM_WORKERS,
#                          shuffle=True)

# model
model = ResNet.resnet32(num_classes=params.NUM_CLASSES).to(params.DEVICE)
# simplification: we initialize the network with all the classes 
optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                            weight_decay=params.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                 gamma=params.GAMMA)  # allow to change the LR at predefined epochs

# Run
exemplars = [None] * params.NUM_CLASSES

test_indexes = []
accs = []
for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):
    train_indexes = train_dataset.get_indexes_groups(task)
    test_indexes = test_indexes + test_dataset.get_indexes_groups(task)

    train_subset = Subset(train_dataset, train_indexes, transform=train_transformer)
    test_subset = Subset(test_dataset, test_indexes, transform=test_transformer)

    train_loader = DataLoader(train_subset,  # num_workers=params.NUM_WORKERS,
                              batch_size=params.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset,  # num_workers=params.NUM_WORKERS,
                             batch_size=params.BATCH_SIZE, shuffle=True)

    model, exemplars = icarl.incremental_train(train_dataset, model, exemplars, task, train_transformer)

    col = []
    for i, x in enumerate(train_splits[:int(task / params.NUM_CLASSES * params.TASK_SIZE)+1]):
        v = np.array(x)
        col = np.concatenate((col, v), axis=None)
        col = col.astype(int)
    mean = None
    total = 0.0
    running_corrects = 0.0
    for img, lbl, _ in train_loader:
        img = img.float().to(params.DEVICE)
        preds, mean = icarl.classify(img, exemplars, model, task, train_dataset, mean)
        preds = preds.to(params.DEVICE)
        labels = utils.map_splits(lbl, col).to(params.DEVICE)

        total += len(lbl)
        running_corrects += torch.sum(preds == labels.data).data.item()

    accuracy = float(running_corrects / total)
    print(f'task: {task}', f'train accuracy = {accuracy}')
    accs.append(accuracy)

    total = 0.0
    running_corrects = 0.0
    tot_preds = []
    tot_lab = []
    for img, lbl, _ in test_loader:
        img = img.float().to(params.DEVICE)
        preds, _ = icarl.classify(img, exemplars, model, task, train_dataset, mean)
        preds = preds.to(params.DEVICE)
        labels = utils.map_splits(lbl, col).to(params.DEVICE)

        tot_preds = np.concatenate((tot_preds, preds.data.cpu().numpy()))
        tot_lab = np.concatenate((tot_lab, labels.data.cpu().numpy()))

        total += len(lbl)
        running_corrects += torch.sum(preds == labels.data).data.item()

    accuracy = float(running_corrects / total)
    print(f'task: {task}', f'test accuracy = {accuracy}')
    cf = confusion_matrix(tot_lab, tot_preds)
    df_cm = pd.DataFrame(cf, range(task + params.TASK_SIZE), range(task + params.TASK_SIZE))
    # sn.set(font_scale=.5)  # for label size
    # sn.heatmap(df_cm, annot=False)
    # plt.show()
