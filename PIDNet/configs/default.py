# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pidnet_s'
_C.MODEL.PRETRAINED = 'pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar'
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 2


_C.LOSS = CN()
_C.LOSS.USE_OHEM = True
_C.LOSS.USE_DICE = False
_C.LOSS.USE_FOCAL = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]
_C.LOSS.SB_WEIGHTS = 0.5

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = 'data/'
_C.DATASET.DATASET = 'loveDa'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/loveDa/train.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/loveDa/val.lst'
_C.DATASET.TARGET_SET = 'list/loveDa/val.lst'

_C.DATASET.SOURCE_DATASET = 'loveDA-Urban'
_C.DATASET.TARGET_DATASET = 'loveDA-Rural'
_C.DATASET.SOURCE_TRAIN_SET = 'list/loveDA-Urban/train.lst'
_C.DATASET.SOURCE_TEST_SET = 'list/loveDA-Urban/val.lst'
_C.DATASET.TARGET_TRAIN_SET = 'list/loveDA-Rural/train.lst'
_C.DATASET.TARGET_TEST_SET = 'list/loveDA-Rural/val.lst'


# training
_C.TRAIN = CN()
_C.TRAIN.IMAGE_SIZE = [1024, 1024]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001
_C.TRAIN.SCHEDULER = False
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1


# Enable Augmentation in general
_C.TRAIN.AUG = False 


#Specific Augmentations
_C.TRAIN.AUG1 = False
_C.TRAIN.AUG2 = False
_C.TRAIN.AUG3 = False
_C.TRAIN.AUG4 = False


_C.TRAIN.AUG_CHANCE = False # Must be used with TRAIN_AUG = True
_C.TRAIN.EVAL_INTERVAL = 1

_C.TRAIN.ADVERSARIAL = False
_C.TRAIN.D1 = False
_C.TRAIN.LR_D1 = 0.001
_C.TRAIN.LR_D2 = 0.001
_C.TRAIN.GAN = 'Vanilla'
_C.TRAIN.LAMBDA_ADV1 = 0.001
_C.TRAIN.LAMBDA_ADV2 = 0.001
_C.TRAIN.LAMBDA_SEG2 = 0.001


_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()
_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False

_C.TEST.OUTPUT_INDEX = -1



def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

