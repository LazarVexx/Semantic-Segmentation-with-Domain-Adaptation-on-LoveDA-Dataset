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


def train_adv(config, epoch, num_epoch, 
          epoch_iters, base_lr, num_iters,
          trainloader, targetloader, 
          optimizer, optimizer_D1, optimizer_D2, 
          model, model_D1, model_D2, 
          writer_dict):
    
    model.train()
    model_D1.train()
    model_D2.train()

    bce_loss = torch.nn.MSELoss() if config.TRAIN.GAN == 'LS' else torch.nn.BCEWithLogitsLoss()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, (batch, target_batch) in enumerate(zip(trainloader, targetloader)):
        # 1. Training Generator
        optimizer.zero_grad()

        # Initialize loss values
        loss_seg_value1 = 0
        loss_seg_value2 = 0
        loss_adv_target_value1 = 0
        loss_adv_target_value2 = 0

        # Freeze discriminator
        for param in model_D1.parameters():
            param.requires_grad = False
        for param in model_D2.parameters():
            param.requires_grad = False

        # Train with source domain
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        h, w = labels.size(1), labels.size(2)

        losses_source, pred_source, acc_source, loss_list_source = model(images, labels, bd_gts)

        loss_seg_1 = losses_source.mean() 
        loss_seg_2 = loss_list_source[2].mean()
        loss_seg = loss_seg_1 + config.TRAIN.LAMBDA_SEG2 * loss_seg_2
        loss_seg.backward()
        loss_seg_value1 += loss_seg_1.item()
        loss_seg_value2 += loss_seg_2.item()

        # Train with target domain
        images_target, _, _, _, _ = target_batch
        images_target = images_target.cuda()

        _, pred_target, _, _ = model(images_target, labels, bd_gts)


        # Adversarial loss on target
        D_out_target_conv5 = model_D1(F.softmax(pred_target[1], dim=1))
        D_out_target_conv4 = model_D2(F.softmax(pred_target[2], dim=1))

        loss_adv_target1 = config.TRAIN.LAMBDA_ADV1 * bce_loss(D_out_target_conv5, 
                                                            torch.ones_like(D_out_target_conv5).cuda())
        loss_adv_target2 = config.TRAIN.LAMBDA_ADV2 * bce_loss(D_out_target_conv4,
                                                            torch.ones_like(D_out_target_conv4).cuda())

        # Normalize and accumulate adversarial losses
        loss_adv = (loss_adv_target1 + loss_adv_target2) 
        loss_adv.backward()
        loss_adv_target_value1 += loss_adv_target1.item() 
        loss_adv_target_value2 += loss_adv_target2.item() 


        # 2. Training Discriminators
        for param in model_D1.parameters():
            param.requires_grad = True
        for param in model_D2.parameters():
            param.requires_grad = True

        # Initialize loss values
        loss_D_value1 = 0
        loss_D_value2 = 0

        # Train with source
        pred_source_conv5 = pred_source[1].detach()
        pred_source_conv4 = pred_source[2].detach()

        D_out_source_conv5 = model_D1(F.softmax(pred_source_conv5, dim=1))
        D_out_source_conv4 = model_D2(F.softmax(pred_source_conv4, dim=1))

        loss_D1_source = bce_loss(D_out_source_conv5, torch.ones_like(D_out_source_conv5).cuda())
        loss_D2_source = bce_loss(D_out_source_conv4, torch.ones_like(D_out_source_conv4).cuda())

        # Train with target
        pred_target_conv5 = pred_target[1].detach()
        pred_target_conv4 = pred_target[2].detach()

        D_out_target_conv5 = model_D1(F.softmax(pred_target_conv5, dim=1))
        D_out_target_conv4 = model_D2(F.softmax(pred_target_conv4, dim=1))

        loss_D1_target = bce_loss(D_out_target_conv5, torch.zeros_like(D_out_target_conv5).cuda())
        loss_D2_target = bce_loss(D_out_target_conv4, torch.zeros_like(D_out_target_conv4).cuda())

        # Combine and normalize losses
        optimizer_D1.zero_grad()
        loss_D1 = (loss_D1_source + loss_D1_target) / (2)
        loss_D1.backward()
        loss_D_value1 += loss_D1.item()
        optimizer_D1.step()

        optimizer_D2.zero_grad()
        loss_D2 = (loss_D2_source + loss_D2_target) / (2)
        loss_D2.backward()
        loss_D_value2 += loss_D2.item()
        optimizer_D2.step()


        # Metrics update
        batch_time.update(time.time() - tic)
        tic = time.time()

        
        optimizer.step()
        ave_loss.update(loss_seg.item())
        ave_acc.update(acc_source.item())
        avg_sem_loss.update(loss_list_source[0].mean().item())
        
        lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Loss_D1: {:.6f}, Loss_D2: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], 
                      ave_loss.average(), loss_D1.item(), loss_D2.item(),
                      ave_acc.average(), avg_sem_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
    
def validate_adv(config, testloader, model, writer_dict):
    model.eval()
    inference_times = []
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            
            start_time = time.time()
            losses, pred, _, _ = model(image, label, bd_gts)
            inference_time = (time.time() - start_time) / config.TEST.BATCH_SIZE_PER_GPU
            inference_times.append(inference_time)
            
            pred = pred[:2]

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array[1:].mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))
    inference_time = np.mean(inference_times)
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array,inference_time

