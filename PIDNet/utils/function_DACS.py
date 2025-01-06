# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from tools.train_DACS import generate_pseudo_labels
from utils.criterion import CrossEntropy, OhemCrossEntropy

def train(config, epoch, num_epochs, source_loader, target_loader, optimizer, model, writer_dict, augmentations):
    model.train()  # Ensure model is in training mode

    total_loss = 0.0
    source_loss_total = 0.0
    target_loss_total = 0.0

    for i, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):
        # Source domain data
        source_images, source_labels = source_data
        source_images, source_labels = source_images.cuda(), source_labels.cuda()

        # Forward pass on source domain
        source_logits = model(source_images)
        source_loss = CrossEntropy()(source_logits, source_labels)

        # Target domain data with strong augmentations
        target_images, _ = target_data  # Ignore target labels
        target_images = target_images.cuda()
        augmented_target_images = augmentations(target_images)

        # Generate pseudo-labels for target domain
        pseudo_labels, confidence_mask = generate_pseudo_labels(model, augmented_target_images)
        pseudo_labels = pseudo_labels[confidence_mask]
        confident_images = augmented_target_images[confidence_mask]

        if confident_images.size(0) > 0:  # Ensure valid pseudo-labels
            target_logits = model(confident_images)
            target_loss = CrossEntropy()(target_logits, pseudo_labels)
        else:
            target_loss = torch.tensor(0.0, requires_grad=True).cuda()

        # Total loss with weighting
        total_batch_loss = source_loss + config.TRAIN.TARGET_LOSS_WEIGHT * target_loss
        total_loss += total_batch_loss.item()
        source_loss_total += source_loss.item()
        target_loss_total += target_loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()

        # Log training progress
        if i % config.LOG_FREQ == 0:
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{len(source_loader)}, "
                         f"Source Loss: {source_loss.item():.4f}, Target Loss: {target_loss.item():.4f}")

    avg_source_loss = source_loss_total / len(source_loader)
    avg_target_loss = target_loss_total / len(target_loader)
    avg_total_loss = total_loss / len(source_loader)

    return {
        'source_loss': avg_source_loss,
        'target_loss': avg_target_loss,
        'total_loss': avg_total_loss
    }



def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            # Unpack batch and move tensors to GPU
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)
            bd_gts = bd_gts.float().cuda(non_blocking=True)

            # Forward pass
            losses, pred, pseudo_label, confidence_mask = model(image, label, bd_gts)

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
        mean_IoU = IoU_array.mean()

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

    writer_dict['valid_global_steps'] = global_steps + 1

    # Return validation metrics
    return ave_loss.average(), mean_IoUs, IoU_arrays


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
