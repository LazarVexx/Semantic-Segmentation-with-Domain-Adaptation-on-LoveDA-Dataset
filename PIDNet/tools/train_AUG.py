# ------------------------------------------------------------------------------ 
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation 
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import torch.optim as optim

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function_AUG import train, validate
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warm-up for learning rate."""
    lr = base_lr * (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Warm-up Epoch {epoch + 1}: Learning Rate = {lr}")


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                         thres=config.LOSS.OHEMTHRES,
                                         min_kept=config.LOSS.OHEMKEEP,
                                         weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=train_dataset.class_weights)

    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV)
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, 
                               weight_decay=config.TRAIN.WD)
    else:
        raise ValueError('Only SGD and Adam optimizers are supported')

    epoch_iters = len(trainloader)
    best_mIoU = 0
    last_epoch = 0

    warmup_epochs = 5
    base_lr = config.TRAIN.LR
    if config.TRAIN.SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(config.TRAIN.END_EPOCH - warmup_epochs), eta_min=1e-6)

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        if epoch < warmup_epochs:
            adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr)
        elif config.TRAIN.SCHEDULER:
            scheduler.step()

        train(config, epoch, config.TRAIN.END_EPOCH, epoch_iters,
              optimizer.param_groups[0]['lr'], len(trainloader),
              trainloader, optimizer, model, writer_dict)

        if epoch % config.TRAIN.EVAL_INTERVAL == 0 or epoch == config.TRAIN.END_EPOCH - 1:
            valid_loss, mean_IoU, IoU_array = validate(
                config, testloader, model, writer_dict)

            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.state_dict(),
                           os.path.join(final_output_dir, 'best_model.pth'))

            logger.info(f'Epoch {epoch + 1}: Loss={valid_loss}, Mean IoU={mean_IoU}')

        checkpoint_path = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        torch.save({'epoch': epoch + 1,
                    'best_mIoU': best_mIoU,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, checkpoint_path)

    writer_dict['writer'].close()
    logger.info('Training completed')
    torch.save(model.state_dict(),
               os.path.join(final_output_dir, 'final_model.pth'))


if __name__ == '__main__':
    main()
