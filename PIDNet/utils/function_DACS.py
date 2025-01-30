# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time
from configs import config
from configs import update_config
import numpy as np
from tqdm import tqdm
import models
import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import visualize_images,visualize_segmentation
from utils.criterion import CrossEntropy, OhemCrossEntropy


def classmix_fn(source_images, source_labels, target_images, pseudo_labels, source_bd_gts, target_bd_gts):
    batch_size, _, height, width = source_images.size()

    # Clone inputs for mixed outputs
    mixed_images = target_images.clone()
    mixed_labels = pseudo_labels.clone()
    mixed_bd_gts = target_bd_gts.clone()

    for i in range(batch_size):
        # Select random classes from the source domain to mix
        unique_classes = source_labels[i].unique()
        num_classes = len(unique_classes)
        selected_classes = unique_classes[torch.randperm(num_classes)[:num_classes // 2]]

        # Create a mask for the selected classes
        class_mask = torch.zeros_like(source_labels[i], dtype=torch.bool)
        for cls in selected_classes:
            class_mask |= source_labels[i] == cls

         #visualize_segmentation(class_mask)

        # Apply the class mask to copy pixels from source to target
        mixed_images[i, :, class_mask] = source_images[i, :, class_mask]
        mixed_labels[i, class_mask] = source_labels[i, class_mask]
        mixed_bd_gts[i, class_mask] = source_bd_gts[i, class_mask]

    return mixed_images, mixed_labels, mixed_bd_gts
   
    

def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, source_loader, target_loader, optimizer, model, writer_dict, criterion):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model_target = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
    
    model = model.to(device)
    model_target = model_target.to(device)

    source_loss_total = 0
    target_loss_total = 0

    for i_iter, (source_data, target_data) in enumerate(zip(source_loader, target_loader), 0):
        # --- Load source domain data ---

        source_images = source_data[0]
        source_labels = source_data[1]
        source_bd_gts = source_data[2]
       
        source_images, source_labels, source_bd_gts = source_images.cuda(), source_labels.long().cuda(), source_bd_gts.float().cuda()

        #visualize_images(source_images[0])
        #visualize_segmentation(source_labels[0])

        # --- Compute source loss ---
        source_model = model(source_images, source_labels, source_bd_gts)
        source_loss, _, source_acc, _ = source_model
        source_loss = source_loss.mean()
        source_loss_total += source_loss.item()
        # Update avg_sem_loss with source loss
        avg_sem_loss.update(source_loss.item())
           
        # --- Load target domain data ---
        target_images = target_data[0] # Ignore target labels
        target_labels = target_data[1]
        target_bd_gts = target_data[2]
        target_images, target_labels, target_bd_gts = target_images.cuda(), target_labels.long().cuda(), target_bd_gts.float().cuda()
        

        # --- Apply augmentations to target domain ---
        #augmented_target_images = augmentations(target_images)

        # --- Generate pseudo-labels for target domain ---
        
        with torch.no_grad():
            target_logits = model_target(target_images)
            upsampled_logits = torch.nn.functional.interpolate(target_logits[1], size=(config.TRAIN.IMAGE_SIZE[0],config.TRAIN.IMAGE_SIZE[1]), mode='bilinear', align_corners=False)
            pseudo_labels = torch.argmax(upsampled_logits, dim=1)      
            
        pseudo_labels = pseudo_labels.long().cuda()
        
        target_logits = model(target_images, pseudo_labels, target_bd_gts)
        target_loss, _, target_acc, _, = target_logits 
        
        target_loss_total += target_loss.mean().item()

        # Update avg_sem_loss with target loss
        avg_bce_loss.update(target_loss.mean().item())


        #visualize_images(target_images[0])
        #visualize_segmentation(pseudo_labels[0])

        # --- Apply MixUp augmentation between source and target images ---
        mixed_images, mixed_labels, mixed_bd_gts = classmix_fn(source_images, source_labels, target_images, pseudo_labels, source_bd_gts, target_bd_gts)
        mixed_images, mixed_labels, mixed_bd_gts = mixed_images.cuda(), mixed_labels.long().cuda(), mixed_bd_gts.float().cuda()
        mixed_model = model(mixed_images, mixed_labels, mixed_bd_gts)
        mixup_loss, _, mixup_acc, _ = mixed_model 
        mixup_loss = mixup_loss.mean()
        
        
        #visualize_images(mixed_images[0])
        #visualize_segmentation(mixed_labels[0])

        # --- Compute total loss ---
        mixup_loss_weight = 0.5

        loss_value = source_loss + mixup_loss_weight*mixup_loss
        
        # --- Measure average accuracy ---
        acc = source_acc + mixup_loss_weight * mixup_acc # Averaging source, target, and mixup accuracy

        # --- Measure elapsed time ---
        batch_time.update(time.time() - tic)
        tic = time.time()

        # --- Update average loss ---
        ave_loss.update(loss_value.mean().item())
        ave_acc.update(acc.item())
        
        # --- Backpropagation and optimization ---
        model.zero_grad()       
        loss_value.backward()
        optimizer.step()
        
        # --- Log training progress ---
        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Source Loss: {:.6f}, Target Loss: {:.6f}, MixUp Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(), mixup_loss.mean().item())
            logging.info(msg)

    # --- Update Tensorboard ---
    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer.add_scalar('train_acc', ave_acc.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    # Return training metrics as a dictionary
    train_metrics = {
        'source_loss': source_loss_total / epoch_iters,
        'target_loss': target_loss_total / epoch_iters,
        'total_loss': ave_loss.average(),
        'accuracy': ave_acc.average()
    }
    
    return train_metrics



def validate(config, testloader, model, writer_dict):
    model.eval()
    inference_times = []
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    
    total_correct_pixels = 0
    total_pixels = 0
    total_class_correct = np.zeros(config.DATASET.NUM_CLASSES)
    total_class_pixels = np.zeros(config.DATASET.NUM_CLASSES)
    
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            # Unpack batch and move tensors to GPU
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            
            start_time = time.time()
            losses, pred, _, _ = model(image, label, bd_gts)
            inference_time = (time.time() - start_time) / config.TEST.BATCH_SIZE_PER_GPU
            inference_times.append(inference_time)

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            
            for i, x in enumerate(pred):
                # Upsample prediction to match label size
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                # Update confusion matrix
                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )
                if idx % 10 == 0:
                    print(idx)
                
                # Calculate pixel accuracy and per-class accuracy
                total_correct_pixels += (x.argmax(1) == label).sum().item()
                total_pixels += label.numel()
                
                for c in range(config.DATASET.NUM_CLASSES):
                    total_class_pixels[c] += (label == c).sum().item()
                    total_class_correct[c] += ((x.argmax(1) == label) & (label == c)).sum().item()

            # Compute loss and update the average
            loss = losses.mean()
            ave_loss.update(loss.item())

    # Compute IoU metrics for each output

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array[1:].mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
      
    
        
    # Calculate Pixel Accuracy and Mean Accuracy
    pixel_acc = total_correct_pixels / total_pixels
    mean_acc = (total_class_correct / np.maximum(1.0, total_class_pixels)).mean()
    inference_time = np.mean(inference_times)
    # Log validation results to TensorBoard
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_pixel_acc', pixel_acc, global_steps)
    writer.add_scalar('valid_mean_acc', mean_acc, global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

    # Return validation metrics
    return mean_IoU, IoU_array, pixel_acc, mean_acc, inference_time



