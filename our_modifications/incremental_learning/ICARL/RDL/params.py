import random

DEVICE = 'cuda'  # 'cuda' or 'cpu'
BATCH_SIZE = 128

NUM_CLASSES = 100  # CIFAR100
NUM_TASKS = 10  # number of batches data is splitted
TASK_SIZE = int(NUM_CLASSES/NUM_TASKS)  # size of each task

NUM_EPOCHS = 200

WEIGHT_DECAY = 0.00001
LR = 0.7
STEP_SIZE = [60,120,160]
GAMMA = 0.2

SEED = 42

NUM_WORKERS = 2

K = 2000
MOMENTUM = 0.9
NETWORK_DEEP = 7
