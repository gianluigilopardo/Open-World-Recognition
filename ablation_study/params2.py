import random

DEVICE = 'cuda'  # 'cuda' or 'cpu'
BATCH_SIZE = 128

NUM_CLASSES = 100  # CIFAR100
NUM_TASKS = 10  # number of batches data is splitted
TASK_SIZE = int(NUM_CLASSES/NUM_TASKS)  # size of each task

NUM_EPOCHS = 160

WEIGHT_DECAY = 5e-4
LR = 0.1
STEP_SIZE = [80, 120]
GAMMA = 0.1

SEED = 42

NUM_WORKERS = 2

K = 2000
MOMENTUM = 0.9
