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
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
 


    for i_iter, batch in enumerate(trainloader, 0):

        
        images, labels, bd_gts, _,_  = batch
        
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        model.cuda()

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc  = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())
        
        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)
        

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
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

            losses, pred, _, _ = model(image, label, bd_gts)
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

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


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
 
def train_adv_D1(config, epoch, num_epoch, 
          epoch_iters, base_lr, num_iters,
          trainloader, targetloader, 
          optimizer, optimizer_D1,
          model, model_D1,
          writer_dict):
    
    # Modalità training
    model.train()
    model_D1.train()

    # Loss function
    if config.TRAIN.GAN == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif config.TRAIN.GAN == 'LS':
        bce_loss = torch.nn.MSELoss()

    # Metriche
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, (batch, target_batch) in enumerate(zip(trainloader, targetloader)):
        # 1. Training del generatore
        optimizer.zero_grad()
        
        # Source domain
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        losses_source, pred_source, acc_source, loss_list_source = model(images, labels, bd_gts)
        loss_seg = losses_source.mean()

        # Target domain
        images_target, labels_target, bd_gts_target, _, _ = target_batch
        images_target = images_target.cuda()
        labels_target = labels_target.long().cuda()
        bd_gts_target = bd_gts_target.float().cuda()
        
        _, pred_target, _, _ = model(images_target, labels_target, bd_gts_target)

        # Loss adversarial
        D_out_target_conv5 = model_D1(F.softmax(pred_target[1].detach(), dim=1))
        
        loss_adv = config.TRAIN.LAMBDA_ADV1 * bce_loss(D_out_target_conv5, 
                                                      torch.ones_like(D_out_target_conv5).cuda())
        
        # Loss totale
        loss = loss_seg + loss_adv
        loss.backward()
        optimizer.step()

        # 2. Training del discriminatore
        optimizer_D1.zero_grad()
        
        # Source domain (label=1)
        D_out_source_conv5 = model_D1(F.softmax(pred_source[1].detach(), dim=1))
        loss_D_source = bce_loss(D_out_source_conv5, torch.ones_like(D_out_source_conv5).cuda())
        
        # Target domain (label=0)
        loss_D_target = bce_loss(D_out_target_conv5, torch.zeros_like(D_out_target_conv5).cuda())
        
        # Loss totale discriminatore
        loss_D = (loss_D_source + loss_D_target) * 0.5
        loss_D.backward()
        optimizer_D1.step()

        # Metriche e logging
        batch_time.update(time.time() - tic)
        ave_loss.update(loss.item())
        ave_acc.update(acc_source.item())
        avg_sem_loss.update(loss_list_source[0].mean().item())
        
        lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Loss_D: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], 
                      ave_loss.average(), loss_D.item(),
                      ave_acc.average(), avg_sem_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1



def train_adv(config, epoch, num_epoch, 
          epoch_iters, base_lr, num_iters,
          trainloader, targetloader, 
          optimizer, optimizer_D1, optimizer_D2, 
          model, model_D1, model_D2, 
          writer_dict):
    
    # Modalità training
    model.train()
    model_D1.train()
    model_D2.train()

    # Loss function
    if config.TRAIN.GAN == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif config.TRAIN.GAN == 'LS':
        bce_loss = torch.nn.MSELoss()

    # Metriche
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
        
        # Source domain training
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        losses_source, pred_source, acc_source, loss_list_source = model(images, labels, bd_gts)
        loss_seg = losses_source.mean()

        # Target domain training
        images_target, _, _, _, _ = target_batch
        images_target = images_target.cuda()
        
        with torch.no_grad():
            _, pred_target, _, _ = model(images_target, labels, bd_gts)


        # Disable discriminator gradients
        for param in model_D1.parameters():
            param.requires_grad = False
        for param in model_D2.parameters():
            param.requires_grad = False

        # Adversarial loss on target
        D_out_target_conv5 = model_D1(F.softmax(pred_target[1], dim=1))
        D_out_target_conv4 = model_D2(F.softmax(pred_target[2], dim=1))
        
        loss_adv_target1 = config.TRAIN.LAMBDA_ADV1 * bce_loss(D_out_target_conv5, 
                                                              torch.ones_like(D_out_target_conv5).cuda()).mean()
        loss_adv_target2 = config.TRAIN.LAMBDA_ADV2 * bce_loss(D_out_target_conv4,
                                                              torch.ones_like(D_out_target_conv4).cuda()).mean()
        
        total_loss = loss_seg + (loss_adv_target1 + loss_adv_target2)
        total_loss.backward()
        optimizer.step()


        # 2. Training Discriminators
        for param in model_D1.parameters():
            param.requires_grad = True
        for param in model_D2.parameters():
            param.requires_grad = True

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()

        # Train on source domain
        pred_source_conv5 = pred_source[1].detach()
        pred_source_conv4 = pred_source[2].detach()
        
        D_out_source_conv5 = model_D1(F.softmax(pred_source_conv5, dim=1))
        D_out_source_conv4 = model_D2(F.softmax(pred_source_conv4, dim=1))

        # Train on target domain
        pred_target_conv5 = pred_target[1].detach()
        pred_target_conv4 = pred_target[2].detach()
        
        D_out_target_conv5 = model_D1(F.softmax(pred_target_conv5, dim=1))
        D_out_target_conv4 = model_D2(F.softmax(pred_target_conv4, dim=1))
        
        # Source domain
        loss_D1_source = bce_loss(D_out_source_conv5, torch.ones_like(D_out_source_conv5).cuda()).mean()
        loss_D2_source = bce_loss(D_out_source_conv4, torch.ones_like(D_out_source_conv4).cuda()).mean()

        # Target domain
        loss_D1_target = bce_loss(D_out_target_conv5, torch.zeros_like(D_out_target_conv5).cuda()).mean()
        loss_D2_target = bce_loss(D_out_target_conv4, torch.zeros_like(D_out_target_conv4).cuda()).mean()

        # Combine losses
        total_loss_D1 = (loss_D1_source + loss_D1_target)
        total_loss_D2 = (loss_D2_source + loss_D2_target)

        total_loss_D1.backward()
        total_loss_D2.backward()
        optimizer_D1.step()
        optimizer_D2.step()

        # Metriche e logging
        batch_time.update(time.time() - tic)
        ave_loss.update(total_loss.item())
        ave_acc.update(acc_source.item())
        avg_sem_loss.update(loss_list_source[0].mean().item())
        
        lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Loss_D1: {:.6f}, Loss_D2: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], 
                      ave_loss.average(), total_loss_D1,total_loss_D2,
                      ave_acc.average(), avg_sem_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1
