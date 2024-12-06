from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

def train(dataloader, model, criterion, optimizer, scheduler, epoch, device):
    '''
        Run one training epoch
    '''
    model.train()
    epoch_loss = 0
    for batch_idx, (img, label) in tqdm(enumerate(dataloader)):
        img, label = img.to(device), label.to(device)
        output = model(img)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss.mean().item())
        epoch_loss += loss.mean().item()
        
        #print(f"Epoch #{epoch} Batch #{batch_idx}: Loss = {loss.mean().item()}")
        
    print(f"Epoch {epoch}: Training Loss = {epoch_loss/ (batch_idx+1)}") # mean batch loss in epoch

def test(model, dataloader, device):
    '''
    Get DICE score for dataset with model
    '''
    dice = 0
    iou = 0
    epsilon = 1e-10
    model.eval()
    for batch_idx, (img, label) in tqdm(enumerate(dataloader)):
        img, label = img.to(device), label.to(device)
        predict = model(img) # predict image
        predict = torch.argmax(predict, dim=1) # get grayscale prediction
        predict = torch.eye(12, device=predict.device)[predict].permute(0,3,1,2)
        predict = F.interpolate(predict.float(), size=label.shape[2:], mode="nearest-exact")  # Match label size
        
        intersection = (predict * label).sum(dim=(2, 3))  # Sum over spatial dimensions H and W
        union = predict.sum(dim=(2, 3)) + label.sum(dim=(2, 3))  # Sum over spatial dimensions H and W
        dice += ((2.0 * intersection + epsilon) / (union + epsilon)).mean()
        iou += ((intersection + epsilon) / (union - intersection + epsilon)).mean()
        
    m_dice = dice / (batch_idx+1)
    m_iou = iou / (batch_idx+1)
    return m_dice, m_iou

def DICELoss(scores, label):
    '''
    multi-class DICE loss for segmentation
    '''
    epsilon = 1e-10
    scores = F.softmax(scores, dim=1)  # Apply softmax across classes to ensure one prediction per pixel
    
    intersection = (scores * label).sum(dim=(2, 3))  # Sum over spatial dimensions H and W
    union = scores.sum(dim=(2, 3)) + label.sum(dim=(2, 3))  # Sum over spatial dimensions H and W
    # Compute Dice coefficient for each class
    dice_coeff = (2.0 * intersection + epsilon) / (union + epsilon)  # Add epsilon to avoid division by zero
    dice_loss = 1 - dice_coeff  # Dice loss per class
    
    return dice_loss.mean()