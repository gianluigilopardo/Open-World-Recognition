import torch
import numpy as np
import math
from owr.ablation_study import params
from owr.ablation_study import utils
# import params
# import utils
from torch.nn import functional as F

def compute_loss(outputs, old_outputs, onehot_labels, task, train_splits, loss_version='opt1'):
    classes = utils.get_classes(train_splits, task-1)
    if(loss_version=='opt1'):
        criterion = torch.nn.BCEWithLogitsLoss()
        m = torch.nn.Sigmoid() #ultimo layer da aggiungere alla rete per la parte di feature extractor
        outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), \
                                              onehot_labels.to(params.DEVICE)
        if task == 0:
            loss = criterion(input=outputs, target=onehot_labels) #se sono nella prima task allora faccio solo classificazione con BCE
        else:
            target = onehot_labels.clone().to(params.DEVICE)
            target[:, classes] = m(old_outputs[:, classes]).to(params.DEVICE)
            loss = criterion(input=outputs, target=target)
        return loss
    if(loss_version=='opt2'):
        criterion = torch.nn.MSELoss()
        m = torch.nn.Softmax(dim=1)
        if( task == 0):
            outputs = m(outputs)
            loss = criterion(outputs,onehot_labels)
        else:
            target = onehot_labels.clone().to(params.DEVICE)
            target[:, classes] = m(old_outputs[:,classes]).to(params.DEVICE)
            outputs = m(outputs)
            loss = criterion(input=outputs, target=target)
        return loss
        
    if(loss_version=='opt3'):
        classCriterion = torch.nn.CrossEntropyLoss()
        distCriterion = torch.nn.MSELoss()
        if( task == 0):
            loss = classCriterion(outputs, onehot_labels)
        else:
            classLoss = classCriterion(outputs, onehot_labels)
            distLoss = distCriterion(outputs[:, classes], old_outputs[:, classes] )
            loss = classLoss + distLoss 
        return loss
def compute_loss_cosine(outputs, target, features, old_features, task):
    classification_loss = torch.nn.CrossEntropyLoss()
    distillation_loss = torch.nn.CosineEmbeddingLoss()
    cosine_gamma = 5
    ys = torch.tensor([1]*len(outputs)).to(params.DEVICE)
    m = torch.nn.Sigmoid() #ultimo layer da aggiungere alla rete per la parte di feature extractor
    outputs, target, features, old_features = outputs.to(params.DEVICE), target.to(params.DEVICE), \
                                          features.to(params.DEVICE), old_features.to(params.DEVICE)
    if task == 0:
        loss = classification_loss(input=outputs, target=target) #se sono nella prima task allora faccio solo classificazione con BCE
    else:
        gamma = cosine_gamma * math.sqrt(task/params.TASK_SIZE) #da verificare
        class_loss = classification_loss(input=outputs, target=target)

        features = F.normalize(features, p=2, dim=1)
        old_features = F.normalize(old_features, p=2, dim=1)

        dist_loss = distillation_loss(features, old_features, ys)
        dist_loss = gamma*dist_loss
        loss = class_loss + dist_loss
    return loss

def compute_loss_l2(outputs, onehot_labels, features, old_features, task):
        classification_loss = torch.nn.BCEWithLogitsLoss()
        distillation_loss = torch.nn.MSELoss()

        if( task == 0):
            loss = classification_loss(outputs,onehot_labels)
        else:
            class_loss = classification_loss(input=outputs, target=onehot_labels)
            dist_loss = distillation_loss(features, old_features)
            loss = class_loss + dist_loss
        return loss

def compute_loss_l1(outputs, onehot_labels, features, old_features, task):
        classification_loss = torch.nn.BCEWithLogitsLoss()
        distillation_loss = torch.nn.L1Loss()

        if( task == 0):
            loss = classification_loss(outputs,onehot_labels)
        else:
            class_loss = classification_loss(input=outputs, target=onehot_labels)
            dist_loss = distillation_loss(features, old_features)
            loss = class_loss + dist_loss
        return loss

def train_network(classes, model, old_model, optimizer, data_loader, scheduler, task, train_splits,loss_version='opt1'):
    for epoch in range(params.NUM_EPOCHS):
        length = 0
        running_corrects = 0
        for images, labels, idx in data_loader:
            images = images.float().to(params.DEVICE)  #"""convert image into float vector"""
            labels = labels.long().to(params.DEVICE)  # .long() #convert in long
            onehot_labels = torch.eye(params.NUM_CLASSES)[labels].to(params.DEVICE) #serve per la BCELoss
            mapped_labels = utils.map_splits(labels, classes)
            optimizer.zero_grad() #azzero i gradienti
            # features=False : use fully connected layer (see ResNet)
            old_outputs = old_model(images, features=False)
            outputs = model(images, features=False)
            old_features = old_model(images, features=True)
            features = model(images, features=True)
            loss = compute_loss_l2(outputs, onehot_labels, features, old_features, task)
            # cut_outputs take only the first #task outputs: see simplification in main
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            running_corrects += torch.sum(preds == mapped_labels.data).data.item()
            length += len(images)
            loss.backward()
            optimizer.step()
        accuracy = running_corrects / float(length)
        scheduler.step()
        print('Step: ' + str(task) + ", Epoch: " + str(epoch) + ", Loss: " +
              str(loss.item()) + ', Accuracy: ' + str(accuracy))
    return model


def trainLWF(task, train_loader, train_splits):
    print(f'task = {task} ')
    resNet = torch.load('resNet_task' + str(task) + '.pt').train(True) #train the previously stored model
    old_resNet = torch.load('resNet_task' + str(task) + '.pt').train(False) #evaluate the model

    # Define the parameters for traininig:
    optimizer = torch.optim.SGD(resNet.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                                weight_decay=params.WEIGHT_DECAY) #ottimizzo modello
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                     gamma=params.GAMMA)  # allow to change the LR at predefined epochs
   
    current_step = 0

    col = np.array(train_splits[int(task / 10)]).astype(int) #salvo in un vettore le classi apparteneti alla task
    #print("train col = ", col)
    #print("train col = ", col[None, :])
    
    #Train phase
    for epoch in range(params.NUM_EPOCHS):
        lenght = 0
        scheduler.step()  # update the learning rate
        running_corrects = 0

        for images, labels, _ in train_loader:
            images = images.float().to(params.DEVICE)
            labels = labels.to(params.DEVICE)
            # print(labels)
            mappedLabels = utils.map_splits(labels, col)
            # print(mappedLabels)
            onehot_labels = torch.eye(100)[labels].to(
                params.DEVICE)  # it creates the one-hot-encoding list for the labels; needed for BCELoss

            optimizer.zero_grad()  # Zero-ing the gradients

            # Forward pass to the network
            old_outputs = old_resNet(images)  #compute the old outputs to which we will apply distillation loss 
            outputs = resNet(images)  # current outputs
            loss = compute_loss(outputs, old_outputs, onehot_labels, task, train_splits) #loss: classification + distillation

            # Get predictions

            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), col[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1) #the predicted class has the highest score and we take the corresponding index
            # print(preds)

            # Update Corrects
            running_corrects += torch.sum(preds == mappedLabels.data).data.item() #compare the indexes of the predicted classes with the indexes of the true labels
            loss.backward()  # backward pass: computes gradients
            optimizer.step()  # update weights based on accumulated gradients
            current_step += 1
            lenght += len(images)
        # Calculate Accuracy
        accuracy = running_corrects / float(lenght)
        print("At step ", str(task), " and at epoch = ", epoch, " the loss is = ", loss.item(), " and accuracy is = ",
              accuracy)
    torch.save(resNet, 'resNet_task{0}.pt'.format(task + 10))


def testLWF(task, test_loader, test_splits):
    criterion = torch.nn.BCEWithLogitsLoss()
    t_l = 0
    resNet = torch.load('resNet_task' + str(task + 10) + '.pt').eval()  # set Network to evaluation mode
    running_corrects = 0

    col = []
    # in testing phase we evaluate on all the classes seen so far,  (test_splits [ : int(task/10) + 1 ] 
    for i, x in enumerate(test_splits[:int(task / 10) + 1]):
        v = np.array(x)
        col = np.concatenate((col, v), axis=None)
    col = col.astype(int)
    
    tot_preds = []
    tot_lab = []
    
    for images, labels, _ in test_loader: #testing phase
        
        images = images.float().to(params.DEVICE)
        labels = labels.to(params.DEVICE)
        mappedLabels = utils.map_splits(labels, col)
        
        onehot_labels = torch.eye(100)[labels].to(params.DEVICE)     #one-hot-encoding list for the labels for BCELoss

        outputs = resNet(images)
        outputs = outputs.to(params.DEVICE)

        cut_outputs = np.take_along_axis(outputs, col[None, :], axis=1)
        cut_outputs = cut_outputs.to(params.DEVICE) #take the elements of interest
        
        _, preds = torch.max(cut_outputs.data, 1) #the predicted label has the highest score and we take the corresponding index
        
        tot_preds = np.concatenate((tot_preds, preds.data.cpu().numpy()))
        tot_lab = np.concatenate((tot_lab, mappedLabels.data.cpu().numpy()))
        
        # Update Corrects
        running_corrects += torch.sum(preds == mappedLabels.data).data.item() #compare the indexes of the predicted classes with the indexes of the true labels
        t_l += len(images)

    # Calculate Accuracy
    accuracy = running_corrects / float(t_l)

    # Calculate Loss

    loss = criterion(outputs, onehot_labels)
    print('Test Loss: {} Test Accuracy : {}'.format(loss.item(), accuracy))
    # cf = confusion_matrix(tot_lab, tot_preds)
    # df_cm = pd.DataFrame(cf, range(task + params.TASK_SIZE), range(task + params.TASK_SIZE))
    # sn.set(font_scale=.4) # for label size
    # sn.heatmap(df_cm, annot=False)
    # plt.show()
    return (accuracy, loss.item())
