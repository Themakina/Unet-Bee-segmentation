import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import dataset

import model
from utils import load_checkpoint, save_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.unet_model().to(DEVICE)

from torchsummary import summary

LOAD_MODEL = True
summary(model, (3, 512, 512))
##hyper parameters##
LEARNING_RATE = 1e-4
num_epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

for epoch in range(num_epochs):
    loop = tqdm(enumerate(dataset.train_batch), total=len(dataset.train_batch))
    for batch_idx, (data, targets) in loop:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        targets = targets.type(torch.long)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)