import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.autograd.gradcheck
from extension import GaussiansTracer
import wandb


device = torch.device("cuda:0")


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)
