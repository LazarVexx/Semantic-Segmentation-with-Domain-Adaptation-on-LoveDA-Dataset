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
from models.model_utils import Discriminator
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function_ADV import validate_adv , train_adv
from utils.utils_adv import create_logger, FullModel


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

# Function to adjust the learning rate during the warm-up phase
def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warm-up."""
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

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet_adv.get_seg_model(config, imgnet_pretrained=imgnet)

    # init D
    if config.TRAIN.ADVERSARIAL:
        model_D1 = Discriminator(num_classes=config.DATASET.NUM_CLASSES)
        model_D1.train()
        model_D1.cuda()
        optimizer_D1 = optim.Adam(model_D1.parameters(), lr=config.TRAIN.LR_D1, betas=(0.9, 0.99))
        optimizer_D1.zero_grad()
        model_D2 = Discriminator(num_classes=config.DATASET.NUM_CLASSES)
        model_D2.train()
        model_D2.cuda()
        optimizer_D2 = optim.Adam(model_D2.parameters(), lr=config.TRAIN.LR_D2, betas=(0.9, 0.99))
        optimizer_D2.zero_grad()
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
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
    
    if config.TRAIN.ADVERSARIAL:
        target_dataset = eval('datasets.'+config.DATASET.DATASET)(
                            root=config.DATASET.ROOT,
                            list_path=config.DATASET.TARGET_SET,
                            num_classes=config.DATASET.NUM_CLASSES,
                            multi_scale=config.TRAIN.MULTI_SCALE,
                            flip=config.TRAIN.FLIP,
                            ignore_label=config.TRAIN.IGNORE_LABEL,
                            base_size=config.TRAIN.BASE_SIZE,
                            crop_size=crop_size,
                            scale_factor=config.TRAIN.SCALE_FACTOR)

        targetloader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=False,
            drop_last=True)


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
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

    # criterion
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

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, 
                           weight_decay=config.TRAIN.WD)  
    else:
        raise ValueError('Only Support SGD optimizer')
    

    

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    
    warmup_epochs = 5  # Number of warm-up epochs
    base_lr = config.TRAIN.LR
    
    if config.TRAIN.SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(config.TRAIN.END_EPOCH - warmup_epochs), eta_min=1e-6
        )
        
    for epoch in range(last_epoch, real_end):
        
        if config.TRAIN.SCHEDULER:
            if epoch < warmup_epochs:
                adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = config.TRAIN.LR

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        if config.TRAIN.ADVERSARIAL:
            train_adv(config, epoch, config.TRAIN.END_EPOCH, 
                epoch_iters, current_lr, num_iters,
                trainloader,targetloader, 
                optimizer,optimizer_D1,optimizer_D2, model,model_D1,model_D2, 
                writer_dict)
        

        if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (epoch >= real_end - 100):
            valid_loss, mean_IoU, IoU_array,inference_time = validate_adv(config, 
                        testloader, model, writer_dict)
        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}, Inference time: {:.3f} sec '.format(
                    valid_loss, mean_IoU, best_mIoU, inference_time)
        logging.info(msg)
        logging.info(IoU_array)

    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
