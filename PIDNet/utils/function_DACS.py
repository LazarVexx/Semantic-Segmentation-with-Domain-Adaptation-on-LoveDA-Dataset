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
from utils.criterion import CrossEntropy, OhemCrossEntropy

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def visualize_images(images, labels, title, num_images=4):
    """
    Visualize a batch of images and their corresponding labels.

    Args:
        images (torch.Tensor): Images tensor, shape (B, C, H, W).
        labels (torch.Tensor): Labels tensor, shape (B, H, W).
        title (str): Title of the visualization.
        num_images (int): Number of images to display from the batch.
    """
    images = images[:num_images].cpu().numpy()  # Take first `num_images`
    labels = labels[:num_images].cpu().numpy()
    num_images = min(num_images, len(images))

    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    fig.suptitle(title, fontsize=16)

    for i in range(num_images):
        # Image
        img = images[i].transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        # Label
        label = labels[i]
        axes[i, 1].imshow(label, cmap="tab20")  # Use a color map for labels
        axes[i, 1].set_title(f"Label {i}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()



def classmix_fn(source_images, source_labels, target_images, pseudo_labels, source_bd_gts, target_bd_gts):
    """
    Perform ClassMix augmentation for DACS by mixing pixels based on class masks.

    Args:
        source_images (torch.Tensor): Source domain images, shape (B, C, H, W).
        source_labels (torch.Tensor): Source domain labels, shape (B, H, W).
        target_images (torch.Tensor): Target domain images, shape (B, C, H, W).
        pseudo_labels (torch.Tensor): Pseudo-labels for target images, shape (B, H, W).
        source_bd_gts (torch.Tensor): Source boundary ground truths, shape (B, H, W).
        target_bd_gts (torch.Tensor): Target boundary ground truths, shape (B, H, W).

    Returns:
        mixed_images (torch.Tensor): Mixed images, shape (B, C, H, W).
        mixed_labels (torch.Tensor): Mixed labels, shape (B, H, W).
        mixed_bd_gts (torch.Tensor): Mixed boundary ground truths, shape (B, H, W).
    """
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

        # Apply the class mask to copy pixels from source to target
        mixed_images[i, :, class_mask] = source_images[i, :, class_mask]
        mixed_labels[i, class_mask] = source_labels[i, class_mask]
        mixed_bd_gts[i, class_mask] = source_bd_gts[i, class_mask]

    return mixed_images, mixed_labels, mixed_bd_gts


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, source_loader, target_loader, optimizer, model, writer_dict, augmentations):
    
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

        # --- Compute source loss ---
        source_logits = model(source_images, source_labels, source_bd_gts)
        source_loss, _, source_acc, _ = source_logits  
        source_loss_total += source_loss.item()

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

        # --- Compute target loss (only for confident pseudo-labels) ---
        if target_images.size(0) > 0:  # Ensure valid pseudo-labels exist
            target_logits = model(target_images, pseudo_labels, target_bd_gts)
            target_loss, _, target_acc, _,_,_ = target_logits 
        else:
            target_loss = torch.tensor(0.0, requires_grad=True).cuda()
            target_acc = torch.tensor(0.0, device=source_acc.device)
        target_loss_total += target_loss.item()

        # --- Apply MixUp augmentation between source and target images ---
        mixed_images, mixed_labels, mixed_bd_gts = classmix_fn(source_images, source_labels, target_images, pseudo_labels, source_bd_gts, target_bd_gts)
        mixed_images, mixed_labels, mixed_bd_gts = mixed_images.cuda(), mixed_labels.long().cuda(), mixed_bd_gts.float().cuda()
        mixed_logits = model(mixed_images, mixed_labels, mixed_bd_gts)
        mixup_loss, _, mixup_acc, _ = mixed_logits  
        
        # --- Compute total loss ---
        source_loss_weight = 0.5
        target_loss_weight = 0.5
        mixup_loss_weight = 0.5

        source_loss = CrossEntropy()(source_logits, source_labels)
        mixed_loss = CrossEntropy()(mixed_logits, mixed_labels)
        
        loss_value = source_loss + mixup_loss_weight + mixed_loss
        
        # --- Measure average accuracy ---
        acc = (source_acc + target_acc + mixup_acc) / 3  # Averaging source, target, and mixup accuracy

        # --- Measure elapsed time ---
        batch_time.update(time.time() - tic)
        tic = time.time()

        # --- Update average loss ---
        ave_loss.update(loss_value.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(source_loss.item())
        avg_bce_loss.update(target_loss.item())

        # --- Update learning rate ---
        lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter + cur_iters)

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
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(), mixup_loss.item())
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

            # Forward pass
            losses, pred, _, _ = model(image, label, bd_gts)

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
        IoU_array = tp / np.maximum(1.0, pos + res - tp)
        mean_IoU = IoU_array.mean()
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
        
    # Calculate Pixel Accuracy and Mean Accuracy
    pixel_acc = total_correct_pixels / total_pixels
    mean_acc = (total_class_correct / np.maximum(1.0, total_class_pixels)).mean()

    # Log validation results to TensorBoard
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_pixel_acc', pixel_acc, global_steps)
    writer.add_scalar('valid_mean_acc', mean_acc, global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

    # Return validation metrics
    return mean_IoU, IoU_array, pixel_acc, mean_acc



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
