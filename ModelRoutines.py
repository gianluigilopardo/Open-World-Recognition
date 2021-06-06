# This file cointains routines for training and evaluation of the incremental models
import torch
import torch.nn.functional as F

def train_model(model, loss_function, optimizer, scheduler, train_loader,device, num_epochs):
    n_total_steps = len(train_loader)
    print(f"The total number of steps for each epoch will be {n_total_steps}")
    for epoch in range(num_epochs):
        for i, (images, labels, _) in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)
            labels_1h = F.one_hot(labels, num_classes = 100).float()

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % (n_total_steps/4) == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
    return model

def evaluate_model(model, test_loader,device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        # n_class_correct = [0 for i in range(10)]
        # n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        acc = n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
    return acc

def train_model_incremental(model,loss_function,optimizer, scheduler, dtab, device,
                            num_incremental_steps,
                            num_batches):
    pass