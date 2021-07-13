import random
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 100  # CIFAR100
NUM_TASKS = 10  # number of batches data is splitted
TASK_SIZE = int(NUM_CLASSES / NUM_TASKS)  # size of each task
# iCaRL
BATCH_SIZE = 128
NUM_EPOCHS = 70
WEIGHT_DECAY = 0.00001
LR = 0.1  # (2 for iCaRL)
STEP_SIZE = [49, 63]
GAMMA = 1 / 5
K = 2000
MOMENTUM = 0.9
SEED = 42

NUM_WORKERS = 2 if torch.cuda.is_available() else 0
# if 4 colab gives warning : UserWarning: This DataLoader will create 4 worker processes in total.
# Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader
# is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze,
# lower the worker number to avoid potential slowness/freeze if necessary.
#   cpuset_checked))

# BiC parameters
BIAS_LR = 0.1
BIAS_GAMMA = 1 / 100
BIAS_STEP_SIZE = [100, 150, 200]
BIAS_WEIGHT_DECAY = 0.0002
BIAS_NUM_EPOCHS = 250

# Rejection Modification parameters
REJECTION_LR = 0.01
REJECTION_GAMMA = 1 / 100
REJECTION_STEP_SIZE = [50, 100, 150, 200]
REJECTION_WEIGHT_DECAY = 0.0002
REJECTION_NUM_EPOCHS = 250
