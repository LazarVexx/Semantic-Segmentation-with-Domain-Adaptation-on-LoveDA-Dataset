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



def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader_source, trainloader_target,
          optimizer_pidnet, optimizer_discriminator,
          model_pidnet, discriminator, writer_dict):
    # Training
    model_pidnet.train()
    discriminator.train()

    batch_time = AverageMeter()
    ave_loss_seg = AverageMeter()
    ave_loss_adv = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    # Iteratori per i due dataloader
    iter_source = iter(trainloader_source)
    iter_target = iter(trainloader_target)

    for i_iter in range(epoch_iters):
        # Caricamento batch dal dominio sorgente (annotato)
        try:
            images_source, labels_source = next(iter_source)
        except StopIteration:
            iter_source = iter(trainloader_source)
            images_source, labels_source = next(iter_source)

        images_source = images_source.cuda()
        labels_source = labels_source.long().cuda()

        # Caricamento batch dal dominio target (non annotato)
        try:
            images_target = next(iter_target)[0]
        except StopIteration:
            iter_target = iter(trainloader_target)
            images_target = next(iter_target)[0]

        images_target = images_target.cuda()

        # Forward pass sul dominio sorgente
        pred_source = model_pidnet(images_source)

        # Calcolo della perdita di segmentazione
        loss_seg = F.cross_entropy(pred_source, labels_source)

        # Forward pass sul dominio target
        pred_target = model_pidnet(images_target)

        # Passaggio delle predizioni al discriminatore
        disc_pred_target = discriminator(pred_target.detach())
        disc_pred_source = discriminator(pred_source.detach())

        # Perdita del discriminatore (sorgente vs target)
        loss_disc_s = F.binary_cross_entropy_with_logits(disc_pred_source,
                                                         torch.ones_like(disc_pred_source))
        loss_disc_t = F.binary_cross_entropy_with_logits(disc_pred_target,
                                                         torch.zeros_like(disc_pred_target))
        loss_disc_total = loss_disc_s + loss_disc_t

        # Aggiornamento del discriminatore
        optimizer_discriminator.zero_grad()
        loss_disc_total.backward()
        optimizer_discriminator.step()

        # Perdita avversaria (per PIDNet)
        disc_pred_target_adv = discriminator(pred_target)
        loss_adv = F.binary_cross_entropy_with_logits(disc_pred_target_adv,
                                                      torch.ones_like(disc_pred_target_adv))

        # Perdita totale per PIDNet
        total_loss_pidnet = loss_seg + config.LOSS.WEIGHT_ADV * loss_adv

        # Aggiornamento di PIDNet
        optimizer_pidnet.zero_grad()
        total_loss_pidnet.backward()
        optimizer_pidnet.step()

        # Misurazione del tempo e aggiornamento delle metriche
        batch_time.update(time.time() - tic)
        tic = time.time()

        ave_loss_seg.update(loss_seg.item())
        ave_loss_adv.update(loss_adv.item())

        if i_iter % config.PRINT_FREQ == 0:
            msg = f'Epoch: [{epoch}/{num_epoch}] Iter: [{i_iter}/{epoch_iters}], ' \
                  f'Time: {batch_time.average():.2f}, Seg Loss: {ave_loss_seg.average():.6f}, ' \
                  f'Adv Loss: {ave_loss_adv.average():.6f}'
            logging.info(msg)

    writer.add_scalar('train_loss_seg', ave_loss_seg.average(), global_steps)
    writer.add_scalar('train_loss_adv', ave_loss_adv.average(), global_steps)
    writer_dict['train_global_steps'] += 1


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
        mean_IoU = IoU_array.mean()
        
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
