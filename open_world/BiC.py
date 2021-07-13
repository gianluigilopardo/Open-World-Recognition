## Class implementing BiC Method
import torch.nn
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from owr import ResNet
from owr import params
from owr import models
from owr import params
from owr.dataset import *

class BiasCorrectionLayer(nn.Module):
    def __init__(self, current_task): #
        super(BiasCorrectionLayer, self).__init__()
        # current_task is an integer that range from 0 to params.NUM_TASK by 1
        # deterministic initialization
        self.alpha = nn.parameter.Parameter(torch.ones(1))
        self.beta = nn.parameter.Parameter(torch.zeros(1))
        self.current_task = current_task

    def forward(self, x):
        out = x
        if self.current_task > 0:
            new_classes = utils.splitter()[self.current_task] # new classes for this task
            out[:, new_classes] = self.alpha * out[:, new_classes] + self.beta
        return out


#### BiC Metod
class BiC_method(nn.Module):
    def __init__(self, num_classes):
        super(BiC_method, self).__init__()
        self.architecture = ResNet.resnet32(num_classes=num_classes)
        self.bias_correction_modules = nn.ModuleList() # [BiasCorrectionLayer(current_task=0)]
        # initialized with the first Bias Correction
        # This first do nothing since the first batch is not incremental
        self.exemplars = [None] * num_classes # a list that will store the exemplars for each class
        self.m = params.TASK_SIZE # number of new classes at each step
        self.n = 0 # number of current old classes
        for i in range(0,params.NUM_TASKS):
            self.bias_correction_modules.append(BiasCorrectionLayer(i))
            if i == 0:
                # we do not use the first two bias correction
                for param in self.bias_correction_modules[i].parameters():
                    param.requires_grad = False

    def first_stage_forward(self, x, features=False):
        # forward pass for the first stage training
        return self.architecture(x, features)

    def forward(self, x, task):
        out = self.architecture(x)
        out = self.bias_correction_modules[int(task / params.TASK_SIZE)](out)
        return out

    def compute_first_loss(self, outputs, old_outputs, labels, task, train_splits, mode):
        if mode == 'BiC loss':
            classification_criterion = torch.nn.CrossEntropyLoss()
            loss_c = classification_criterion(input = outputs, target = labels)
            # print(f"Classfication Loss : {loss_c}")
            lambda_scalar = self.n / (self.n + self.m)
            # print(f"lambda_scalar : {lambda_scalar}")
            if task == 0:
                loss = loss_c
            if task > 0:
                soft_max = nn.Softmax(dim=1)
                log_soft_max = nn.LogSoftmax(dim=1)
                old_classes = utils.get_classes(train_splits, task - 1)  # old classes
                T = 2  # temperature scalar
                with torch.no_grad():
                    old_outputs_scaled = old_outputs / T
                    pi_k_old = soft_max(old_outputs_scaled[:, old_classes])
                outputs_scaled = outputs / T
                pi_k = log_soft_max(outputs_scaled[:, old_classes])
                loss_d = -torch.mean(torch.sum(pi_k_old * pi_k, dim = 1)) # -torch.mean(torch.sum(pre_p * logp, dim=1))
                # print(f"Distillation Loss : {loss_d}")
                loss = lambda_scalar * loss_d + (1 - lambda_scalar) * loss_c
                # print(f"Overall Loss : {loss}")
        if mode == 'iCaRL loss':
            onehot_labels = torch.eye(params.NUM_CLASSES)[labels].to(params.DEVICE)  # serve per la BCELoss
            ## iCaRL Loss
            criterion = torch.nn.BCEWithLogitsLoss()
            sigmoid = torch.nn.Sigmoid()
            outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), \
                                                  onehot_labels.to(params.DEVICE)
            # print('outputs: ' + str(outputs))
            old_classes = utils.get_classes(train_splits, task - 1)  # old classes
            if task == 0:
                loss = criterion(input=outputs, target=onehot_labels)
            if task > 0:
                target = onehot_labels.clone().to(params.DEVICE)
                target[:, old_classes] = sigmoid(old_outputs[:, old_classes]).to(params.DEVICE)
                loss = criterion(input=outputs, target=target)
        return loss

    def incremental_training(self, train_dataset, train_transformer, task, new_test_loader, test_loader, with_rejection = False):
        self.architecture.train(True)
        final_loss_curve, final_training_accs = [], []
        train_splits = utils.splitter()  # labels indexes of the splits
        train_indexes, corrisponding_training_labels = utils.get_task_indexes_with_labels(train_dataset, task) # indexes of the new images for this task
        classes = utils.get_classes(train_splits, task) # labels considered in thi task
        # standard training loader
        train_subset = Subset(train_dataset, train_indexes, train_transformer)
        train_data_loader = DataLoader(train_subset, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS,
                                       shuffle=True)  # data loader for this task
        if task > 0:
            # Train & Validation Splitting from the third batch on
            # Join new samples with the exemplars
            proportion = 0.1
            train_idx, val_idx = utils.get_train_val_indexes(train_indexes, corrisponding_training_labels, self.exemplars,
                                                         proportion, task)
            # data_idx = utils.get_indexes(train_indexes, self.exemplars)
            train_subset = Subset(train_dataset, train_idx, train_transformer)
            train_data_loader = DataLoader(train_subset, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS,
                                       shuffle=True)  # data loader for this task
            val_subset = Subset(train_dataset, val_idx, train_transformer)
            val_data_loader = DataLoader(val_subset, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS,
                                     shuffle=True)  # data loader for this task
        if with_rejection:
            # the dataset change!
            proportion = 0.1
            train_idx, val_idx, val_rejection_idx = utils.get_train_val1_val2_indexes_for_rejection(train_indexes, corrisponding_training_labels,
                                                                                                    self.exemplars, proportion, task)
            train_subset = Subset(train_dataset, train_idx, train_transformer)
            train_data_loader = DataLoader(train_subset, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS,
                                       shuffle=True)  # data loader for this task
            if task > 0:
                val_subset = Subset(train_dataset, val_idx, train_transformer)
                val_data_loader = DataLoader(val_subset, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS,
                                     shuffle=True)  # data loader for this task
            val_rejection_subset = Subset(train_dataset, val_rejection_idx, train_transformer)
            val_rejection_data_loader = DataLoader(val_rejection_subset, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS,
                                     shuffle=True)  # data loader for this task
        print(f"\nFIRST STAGE OF TRAINING, Task : {task}\n")
        # first stage training
        loss_curve, training_accs = self.first_stage_training(train_data_loader, classes, task, train_splits)
        if task > 0:
            # before performing the bias correction we evaluate the baseline
            models.evaluate_model(self, task, new_test_loader, test_loader)
            print(f"\n SECOND STAGE OF TRAINING, Task : {task} \n")
            # second stage training
            second_stage_loss_curve, second_stage_train_accs = self.second_stage_training(val_data_loader, classes, task)
            loss_curve = loss_curve + second_stage_loss_curve
            training_accs = training_accs + second_stage_train_accs
        final_loss_curve = final_loss_curve + loss_curve
        final_training_accs = final_training_accs + training_accs
        # update the exemplar set
        exemplars_per_label = int(params.K / (task + params.TASK_SIZE) + 0.5)  # number of exemplars
        self.construct_exemplar_set(exemplars_per_label, classes[task:], train_dataset, train_indexes)
        self.reduce_exemplars(exemplars_per_label)
        self.n+=params.TASK_SIZE # update the number of old classes
        if not with_rejection:
            val_rejection_data_loader = None
        return final_loss_curve, final_training_accs, val_rejection_data_loader

    def first_stage_training(self, data_loader, classes, task, train_splits):
        optimizer = torch.optim.SGD(self.architecture.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                                          weight_decay=params.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                               gamma=params.GAMMA)  # allow to change the LR at predefined epochs
        loss_curve = []
        train_accs = []
        # self.architecture.train(True)
        # total_params = sum(p.numel() for p in self.architecture.parameters() if p.requires_grad)
        # print("Solver total trainable parameters : ", total_params)
        mode = 'BiC loss' # 'iCaRL loss'
        old_model = copy.deepcopy(self)  # self.architecture? we keep the current (old) model
        # old_model.train(False)  # = .test()
        # old_architecture = copy.deepcopy(self.architecture)  # self.architecture? we keep the current (old) model
        # old_architecture.train(False)  # = .test()
        # old_bias = copy.deepcopy(self.bias_correction_modules[int((task-params.TASK_SIZE)/params.TASK_SIZE)])  # self.architecture? we keep the current (old) model
        # old_bias.train(False)  # = .test()
        for epoch in range(params.NUM_EPOCHS):
            ls = []
            length = 0
            running_corrects = 0
            for images, labels, _ in data_loader:
                images = images.float().to(params.DEVICE)
                labels = labels.long().to(params.DEVICE)  # .long()
                mapped_labels = utils.map_splits(labels, classes)
                optimizer.zero_grad()
                # features=False : use fully connected layer (see ResNet)
                with torch.no_grad():
                    old_outputs = old_model(images, task - params.TASK_SIZE)
                    # old_outputs = old_architecture(images)
                    # old_outputs = old_bias(old_outputs)
                # outputs = self.architecture(images)
                outputs = self.architecture(images) # only train the architecture
                # compute the loss written in the paper
                loss = self.compute_first_loss(outputs, old_outputs, labels, task, train_splits, mode)
                ls.append(loss.item())
                # cut_outputs take only the first #task outputs: see simplification in main
                cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
                _, preds = torch.max(cut_outputs.data, 1)
                running_corrects += torch.sum(preds == mapped_labels.data).data.item()
                length += len(images)
                loss.backward()
                optimizer.step()
            accuracy = running_corrects / float(length)
            train_accs.append(accuracy)
            averaged_loss = np.mean(ls)
            loss_curve.append(averaged_loss)
            scheduler.step() # averaged_loss for reduce on plateau
            print('Step: ' + str(task) + ", Epoch: " + str(epoch) + ", Loss: " +
                  str(averaged_loss) + ', Accuracy: ' + str(accuracy))
        return loss_curve, train_accs

    def second_stage_training(self, data_loader, classes, task):
        # optimizer = torch.optim.SGD(self.bias_correction_modules[int(task / params.TASK_SIZE)].parameters(),
        #                                  lr=0.0001, momentum=params.MOMENTUM,
        #                                  weight_decay=0.1)
        # or lr=0.001, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 60], gamma=0.1)  # allow to change the LR at predefined epochs
        optimizer = torch.optim.Adam(self.bias_correction_modules[int(task / params.TASK_SIZE)].parameters(),
                                     lr=params.BIAS_LR, weight_decay=params.BIAS_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.BIAS_STEP_SIZE, gamma=params.BIAS_GAMMA)
        # activate gradient of bias correction for the stage
        loss_curve = []
        train_accs = []
        # freeze the baseline
        # self.architecture.train(False)
        # Check
        # total_params = sum(p.numel() for p in self.bias_correction_modules[int(task / params.TASK_SIZE)].parameters() if p.requires_grad)
        # print("Solver total trainable parameters : ", total_params)
        classification_criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(params.BIAS_NUM_EPOCHS):
            ls = []
            length = 0
            running_corrects = 0
            for images, labels, _ in data_loader:
                images = images.float().to(params.DEVICE)
                labels = labels.long().to(params.DEVICE)  # .long()
                mapped_labels = utils.map_splits(labels, classes)
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs = self.architecture(images)
                outputs = self.bias_correction_modules[int(task/params.TASK_SIZE)](outputs)
                # compute the loss written in the paper
                loss = classification_criterion(outputs, labels)
                ls.append(loss.item())
                # cut_outputs take only the first #task outputs: see simplification in main
                # print(classes)
                cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
                # print(f"outputs :  {outputs.shape}")
                # print(f"cut outputs :  {cut_outputs.shape}")
                _, preds = torch.max(cut_outputs.data, 1)
                running_corrects += torch.sum(preds == mapped_labels.data).data.item()
                length += len(images)
                loss.backward()
                optimizer.step()
            accuracy = running_corrects / float(length)
            train_accs.append(accuracy)
            averaged_loss = np.mean(ls)
            loss_curve.append(averaged_loss)
            scheduler.step()
            print('Step: ' + str(task) + ", Epoch: " + str(epoch) + ", Loss: " +
                  str(averaged_loss) + ', Accuracy: ' + str(accuracy))
        print(f"\n current_task : {self.bias_correction_modules[int(task / params.TASK_SIZE)].current_task}")
        print(f"alpha : {self.bias_correction_modules[int(task / params.TASK_SIZE)].alpha}")
        print(f"beta : {self.bias_correction_modules[int(task / params.TASK_SIZE)].beta} \n")
        # The bias for this task is then learned, we can freeze it
        for param in self.bias_correction_modules[int(task / params.TASK_SIZE)].parameters():
            param.requires_grad = False
        return loss_curve, train_accs


    # exemplars management algorithms

    # Algorithm 4 iCaRL CONSTRUCT EXEMPLAR SET
    def construct_exemplar_set(self, exemplars_per_label, classes, train_dataset, train_indexes, random_set=False):
        # classes: current classes  # print(classes)
        # exemplars = copy.deepcopy(self.exemplars)
        for image_class in classes:
            images_idx = []
            for i in train_indexes:
                image, label = train_dataset.__getitem__(i)
                if label == image_class:
                    images_idx.append(i)
            if random_set is not True:
                self.exemplars[image_class] = self.generate_new_exemplars(images_idx, exemplars_per_label, train_dataset)
            else:
                self.exemplars[image_class] = random.sample(images_idx, exemplars_per_label)

    # Algorithm 5 iCaRL REDUCE EXEMPLAR SET
    def reduce_exemplars(self, exemplars_per_label):
        exemplars = copy.deepcopy(self.exemplars)
        for i, exemplar_class_i in enumerate(exemplars):
            if exemplar_class_i is not None:
                self.exemplars[i] = exemplar_class_i[:exemplars_per_label]
        return None

    def generate_new_exemplars(self, images_idx, exemplars_per_label, train_dataset):
        features = []
        with torch.no_grad():
            for idx in images_idx:
                image, label = train_dataset.__getitem__(idx)
                image = torch.unsqueeze(image, 0).float()
                x = self.first_stage_forward(image.to(params.DEVICE), features=True).data.cpu().numpy()
                x = x / np.linalg.norm(x)
                features.append(x[0])
        features = np.array(features)
        means = np.mean(features, axis=0)
        means = means / np.linalg.norm(means)
        new_exemplars = []
        phi_new_exemplars = []
        images_idx = copy.deepcopy(images_idx)
        for k in range(0, exemplars_per_label):
            phi_x = features  # features of all images in current class
            phi_P = np.sum(phi_new_exemplars, axis=0)  # sum on all existing exemplars
            mu = 1 / (k + 1) * (phi_x + phi_P)
            exemplar_idx = np.argmin(np.sqrt(np.sum(means - mu) ** 2),
                                     axis=0)  # idx of exemplar with min euclidean distance
            new_exemplars.append(images_idx[exemplar_idx])
            phi_new_exemplars.append(features[exemplar_idx])
            features = np.delete(features, exemplar_idx, axis=0)
            images_idx.pop(exemplar_idx)
        # print(f" Newly added indexes are {new_exemplars}")
        # print(f"with shape {len(new_exemplars)}")
        return new_exemplars
