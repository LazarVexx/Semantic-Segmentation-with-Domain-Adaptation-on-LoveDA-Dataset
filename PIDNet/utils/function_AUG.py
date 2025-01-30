# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import torch.nn as nn
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.autograd import Variable

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from configs import config



def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
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

    for i_iter, batch in enumerate(trainloader, 0):
        # Unpack the batch to handle both original and augmented data
        if config.TRAIN.AUG_RETAIN:
            original_images, original_labels, original_edges, _, _, \
            augmented_images, augmented_labels, augmented_edges, _, _ = batch

            # Move tensors to GPU
            original_images = original_images.cuda()
            original_labels = original_labels.long().cuda()
            original_edges = original_edges.float().cuda()
            
            augmented_images = augmented_images.cuda()
            augmented_labels = augmented_labels.long().cuda()
            augmented_edges = augmented_edges.float().cuda()
            
            # Process original images
            orig_losses, _, orig_acc, orig_loss_list = model(original_images, original_labels, original_edges)
            orig_loss = orig_losses.mean()
            orig_acc = orig_acc.mean()
            
            # Process augmented images
            aug_losses, _, aug_acc, aug_loss_list = model(augmented_images, augmented_labels, augmented_edges)
            aug_loss = aug_losses.mean()
            aug_acc = aug_acc.mean()
            
            # Combine losses
            total_loss = orig_loss + aug_loss
            total_loss.backward()

            # Update metrics
            ave_loss.update(total_loss.item())
            ave_acc.update((orig_acc.item() + aug_acc.item()) / 2)
            avg_sem_loss.update((orig_loss_list[0].mean().item() + aug_loss_list[0].mean().item()) / 2)
            avg_bce_loss.update((orig_loss_list[1].mean().item() + aug_loss_list[1].mean().item()) / 2)
        else:
            # If AUG_RETAIN is False, process as usual
            images, labels, bd_gts, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()
            bd_gts = bd_gts.float().cuda()

            losses, _, acc, loss_list = model(images, labels, bd_gts)
            loss = losses.mean()
            acc = acc.mean()

            loss.backward()

            # Update metrics
            ave_loss.update(loss.item())
            ave_acc.update(acc.item())
            avg_sem_loss.update(loss_list[0].mean().item())
            avg_bce_loss.update(loss_list[1].mean().item())

        # Optimization step
        optimizer.step()
        model.zero_grad()

        # Measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # Skip redundant learning rate adjustment here
        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    # Return training metrics as a dictionary
    train_metrics = {
        'total_loss': ave_loss.average(),
        'accuracy': ave_acc.average(),
        'semantic_loss': avg_sem_loss.average(),
        'bce_loss': avg_bce_loss.average()
    }
    
    return train_metrics





def validate(config, testloader, model, writer_dict):
    model.eval()
    inference_times = []
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            # Print the batch to inspect its structure
            print(f"Batch {idx}: {len(batch)} elements")
            #print(batch)

            # Unpack batch based on its size
            if len(batch) == 5:
                image, label, bd_gts, _, _ = batch
            elif len(batch) == 6:
                image, label, bd_gts, _, _, name = batch
            elif len(batch) == 10:  # If batch has 10 elements
                # Adjust this based on the actual content of your batch
                image, label, bd_gts, _, _, name, *extra_elements = batch
                # You can now inspect extra_elements if needed
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            size = label.size()
            image = image.cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            bd_gts = bd_gts.float().cuda(non_blocking=True)

            # Forward pass
            
            start_time = time.time()
            losses, pred, pseudo_label, confidence_mask = model(image, label, bd_gts)
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
            
            # Compute loss and update the average
            loss = losses.mean().item()
            ave_loss.update(loss)

            if idx % 10 == 0 or idx == len(testloader) - 1:
                logging.info(f'Validation Progress: {idx + 1}/{len(testloader)} batches processed.')

    # Compute IoU metrics for each output
    IoU_arrays = []
    mean_IoUs = []
    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = tp / np.maximum(1.0, pos + res - tp)
        mean_IoU = IoU_array[1:].mean()

        IoU_arrays.append(IoU_array)
        mean_IoUs.append(mean_IoU)

        logging.info(f'Output {i}: IoU per class: {IoU_array}, Mean IoU: {mean_IoU}')

    # Log validation results to TensorBoard
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)

    for i, mean_IoU in enumerate(mean_IoUs):
        writer.add_scalar(f'valid_mIoU_output_{i}', mean_IoU, global_steps)
        writer.add_scalars(
            f'valid_IoU_output_{i}',
            {f'class_{c}': IoU for c, IoU in enumerate(IoU_arrays[i])},
            global_steps
        )
    inference_time = np.mean(inference_times)
    writer_dict['valid_global_steps'] = global_steps + 1

    # Return validation metrics
    return ave_loss.average(), mean_IoUs, IoU_arrays, inference_time


