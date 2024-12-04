from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
import torchvision.transforms.functional as F
import numpy as np

def train(dataloader, model, criterion, optimizer, scheduler, epoch, device):
    '''
        Run one training epoch
    '''

    model.train()
    epoch_loss = 0
    for batch_idx, (img, label) in tqdm(enumerate(dataloader)):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output.logits, label)
        loss.backward()
        optimizer.step()



def test():
    pass

def DICE(model, test_dataloader, device, smooth=1e-10,):
    dice = []
    model.eval()
    for data in test_dataloader:
        img, label = data
        img, label = img.to(device), label.to(device)
        predict = model(img) > 0.5
        num = 2 * (predict * label).sum()
        denum = predict.sum() + label.sum()
        dice.append(((num + smooth) / (denum + smooth)).item())
    m_dice = np.mean(dice)
    return m_dice

def DICELoss(scores, label):
    '''
    multi-class DICE loss for segmentation
    '''
    epsilon = 1e-10
    intersection = (scores * label).sum(dim=(2, 3))  # Sum over spatial dimensions H and W
    union = scores.sum(dim=(2, 3)) + label.sum(dim=(2, 3))  # Sum over spatial dimensions H and W
    
    # Compute Dice coefficient for each class
    dice_coeff = (2.0 * intersection + epsilon) / (union + epsilon)  # Add epsilon to avoid division by zero
    dice_loss = 1 - dice_coeff  # Dice loss per class
    
    return dice_loss.mean()