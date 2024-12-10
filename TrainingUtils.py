from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

def train(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, epoch, device):
    '''
        Run one training epoch
    '''
    model.train()
    epoch_loss = 0
    for batch_idx, (img, label) in tqdm(enumerate(train_dataloader)):
        img, label = img.to(device), label.to(device)
        output = model(img)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        epoch_loss += loss.mean().item()
        
    mdice, miou, mdice_robot, miou_robot = 0, 0, 0, 0
    training_loss = epoch_loss/ (batch_idx+1)
    
    if epoch % 5 == 0:
        mdice, miou, mdice_robot, miou_robot = test(model, val_dataloader, device)
    print(f"Epoch {epoch}: Training Loss = {training_loss}, mDICE: {mdice}, mIoU: {miou} mDICE Robot: {mdice_robot}, mIoU Robot: {miou_robot}") # mean batch loss in epoch and test scores
    
    return [epoch, training_loss, mdice, miou, mdice_robot, miou_robot]

def test(model, dataloader, device):
    '''
    Get DICE score for dataset with model
    '''
    dice = 0
    iou = 0
    dice_robot = 0
    iou_robot = 0
    epsilon = 1e-10
    robot_tools = [1,2,3,8,9,11] # labels that are part or held by robot
    model.eval()
    for batch_idx, (img, label) in tqdm(enumerate(dataloader)):
        img, label = img.to(device), label.to(device)
        predict = model(img) # predict image
        predict = torch.argmax(predict, dim=1) # get grayscale prediction
        predict = torch.eye(12, device=predict.device)[predict].permute(0,3,1,2)
        predict = F.interpolate(predict.float(), size=label.shape[2:], mode="nearest-exact")  # Match label size
        
        # compute DICE and IoU for all labels
        intersection = (predict * label).sum(dim=(2, 3))  # Sum over spatial dimensions H and W
        union = predict.sum(dim=(2, 3)) + label.sum(dim=(2, 3))  # Sum over spatial dimensions H and W
        dice += ((2.0 * intersection + epsilon) / (union + epsilon)).mean()
        iou += ((intersection + epsilon) / (union - intersection + epsilon)).mean()
        
        # compute DICe and IoU for only robot tool labels
        robot_intersection = intersection[:, robot_tools].sum(dim=1)  # Focus only on robot tool channels
        robot_union = union[:, robot_tools].sum(dim=1)
        dice_robot += ((2.0 * robot_intersection + epsilon) / (robot_union + epsilon)).mean()
        iou_robot += ((robot_intersection + epsilon) / (robot_union - robot_intersection + epsilon)).mean()

        
    m_dice = dice / (batch_idx+1)
    m_iou = iou / (batch_idx+1)
    m_dice_robot = dice_robot / (batch_idx + 1)
    m_iou_robot = iou_robot / (batch_idx + 1)
    return m_dice, m_iou, m_dice_robot, m_iou_robot

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