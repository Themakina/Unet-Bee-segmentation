import torch
from torch import nn

from dataset import train_batch, test_batch
from training import DEVICE, model
from utils import load_checkpoint
import matplotlib.pyplot as plt
import numpy as np


LOAD_MODEL = True


def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            print("1")

    print(f"Got {num_correct}//{num_pixels} with acc {num_correct // num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


#check_accuracy(train_batch, model)
#check_accuracy(test_batch, model)
if __name__ == "__main__":
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    for x,y in test_batch:
        x = x.to(DEVICE)
        fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
        softmax = nn.Softmax(dim=1)
        preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
        img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
        preds1 = np.array(preds[0,:,:])
        mask1 = np.array(y[0,:,:])
        img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
        preds2 = np.array(preds[1,:,:])
        mask2 = np.array(y[1,:,:])
        img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
        preds3 = np.array(preds[2,:,:])
        mask3 = np.array(y[2,:,:])
        ax[0,0].set_title('Image')
        ax[0,1].set_title('Prediction')
        ax[0,2].set_title('Mask')
        ax[1,0].set_title('Image')
        ax[1,1].set_title('Prediction')
        ax[1,2].set_title('Mask')
        ax[2,0].set_title('Image')
        ax[2,1].set_title('Prediction')
        ax[2,2].set_title('Mask')
        ax[0][0].axis("off")
        ax[1][0].axis("off")
        ax[2][0].axis("off")
        ax[0][1].axis("off")
        ax[1][1].axis("off")
        ax[2][1].axis("off")
        ax[0][2].axis("off")
        ax[1][2].axis("off")
        ax[2][2].axis("off")
        ax[0][0].imshow(img1)
        ax[0][1].imshow(preds1)
        ax[0][2].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(preds2)
        ax[1][2].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(preds3)
        ax[2][2].imshow(mask3)
        plt.show()

        break
