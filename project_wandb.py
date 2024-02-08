# This is an example of using wandb script for tracking

import wandb
import random

import os

hostname = os.uname()[1]

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="sudoku-project",
    name = hostname + ":" + wandb.util.generate_id(),

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    "monitor_gym": True
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()