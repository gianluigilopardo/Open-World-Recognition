import random
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

NUM_CLASSES = 100  # CIFAR100
NUM_TASKS = 10  # number of batches data is splitted
TASK_SIZE = int(NUM_CLASSES/NUM_TASKS)  # size of each task

NUM_EPOCHS = 100

WEIGHT_DECAY = 0.00001
LR = 2
STEP_SIZE = [20, 40, 60, 80]
GAMMA = 1 / 10

SEED = 42

NUM_WORKERS = 4 if torch.cuda.is_available() else 0

K = 2000
MOMENTUM = 0.9
