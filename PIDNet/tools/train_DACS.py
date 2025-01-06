# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import logging
import timeit
import random

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg', help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml", type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts', help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args


# Function to generate pseudo-labels
def generate_pseudo_labels(model, target_images, confidence_threshold=0.8):
    """Generate pseudo-labels for target domain images."""
    with torch.no_grad():
        logits = model(target_images)
        probs = torch.softmax(logits, dim=1)
        pseudo_labels = torch.argmax(probs, dim=1)
        confidence_mask = probs.max(dim=1).values > confidence_threshold
        return pseudo_labels, confidence_mask


# Function to adjust the learning rate during the warm-up phase
def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warm-up."""
    lr = base_lr * (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Warm-up Epoch {epoch + 1}: Learning Rate = {lr}")


# Strong augmentations for DACS
strong_augmentations = transforms.Compose([
    transforms.RandomResizedCrop(size=(1024, 1024), scale=(0.5, 2.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor()
])


def main():
    args = parse_args()

    if args.seed > 0:
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {'writer': SummaryWriter(tb_log_dir), 'train_global_steps': 0, 'valid_global_steps': 0}

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0

    # Prepare model
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
    model = FullModel(model, CrossEntropy(), BondaryLoss())
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Prepare datasets
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    source_train_dataset = eval('datasets.' + config.DATASET.SOURCE_DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.SOURCE_TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR
    )
    
    source_trainloader = torch.utils.data.DataLoader(
        source_train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True
    )

    target_train_dataset = eval('datasets.' + config.DATASET.TARGET_DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TARGET_TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=(config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]),
        scale_factor=config.TRAIN.SCALE_FACTOR
    )

    target_trainloader = torch.utils.data.DataLoader(
        target_train_dataset,
        batch_size=config.TRAIN.TARGET_BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True
    )

    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.END_EPOCH, eta_min=1e-6)

    # Training Loop
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        model.train()
        for source_data, target_data in zip(source_trainloader, target_trainloader):
            source_imgs, source_labels = source_data
            target_imgs, _ = target_data

            # Generate pseudo-labels
            pseudo_labels, confidence_mask = generate_pseudo_labels(model, target_imgs)
            target_imgs = target_imgs[confidence_mask]
            pseudo_labels = pseudo_labels[confidence_mask]

            # Combine source and target data
            combined_imgs = torch.cat((source_imgs, target_imgs), dim=0)
            

            # Apply augmentations
            combined_imgs = torch.stack([strong_augmentations(img) for img in combined_imgs])

            # Forward pass
            preds = model(combined_imgs)

            # Loss calculation
            source_loss = CrossEntropy()(preds[:len(source_imgs)], source_labels)
            target_loss = CrossEntropy()(preds[len(source_imgs):], pseudo_labels)
            loss = source_loss + 0.5 * target_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Step scheduler
        scheduler.step()

        # Validation
        if epoch % config.PRINT_FREQ == 0 or epoch == config.TRAIN.END_EPOCH - 1:
            validate(config, target_trainloader, model, writer_dict)

    torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'final_state.pt'))
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
