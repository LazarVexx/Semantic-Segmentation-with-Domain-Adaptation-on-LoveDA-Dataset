#!/bin/bash

# PIDNet Small LoveDA 2b
python tools/train.py --cfg configs/loveDa/tests/pidnet_small_loveda_2b.yaml GPUS "[0]"
# PIDNet Small LoveDA 3a
python tools/train.py --cfg configs/loveDa/tests/pidnet_small_loveda_3a.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b
python tools/train.py --cfg configs/loveDa/tests/pidnet_small_loveda_3b.yaml GPUS "[0]"
# PIDNet Small LoveDA Train
python tools/train.py --cfg configs/loveDa/tests/pidnet_small_loveda_train.yaml GPUS "[0]"

# PIDNet Small LoveDA Adam CE
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_adam_ce.yaml GPUS "[0]"
# PIDNet Small LoveDA Adam CE 1024
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_adam_ce_1024.yaml GPUS "[0]"
# PIDNet Small LoveDA Adam CE Scheduler
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_adam_ce_scheduler.yaml GPUS "[0]"
# PIDNet Small LoveDA Adam Dice
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_adam_dice.yaml GPUS "[0]"
# PIDNet Small LoveDA Adam Focal
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_adam_focal.yaml GPUS "[0]"
# PIDNet Small LoveDA Adam OCE
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_adam_oce.yaml GPUS "[0]"
# PIDNet Small LoveDA Adam OCE Scheduler Data Aug Analysis
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_adam_oce_scheduler_data_aug_analysis.yaml GPUS "[0]"
# PIDNet Small LoveDA SDG Dice
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_sdg_dice.yaml GPUS "[0]"
# PIDNet Small LoveDA SDG Focal
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_sdg_focal.yaml GPUS "[0]"
# PIDNet Small LoveDA SGD CE
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_sgd_ce.yaml GPUS "[0]"
# PIDNet Small LoveDA SGD CE Scheduler
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_sgd_ce_scheduler.yaml GPUS "[0]"
# PIDNet Small LoveDA SGD OCE
python tools/train.py --cfg configs/loveDa/tests/2b_test/pidnet_small_loveda_sgd_oce.yaml GPUS "[0]"

# PIDNet Small LoveDA 3b AUG1+AUG2+AUG3
python tools/train.py --cfg configs/loveDa/tests/3b_test/pidnet_small_loveda_3b_AUG1+AUG2+AUG3.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b AUG_CHANCE+AUG1+AUG2
python tools/train.py --cfg configs/loveDa/tests/3b_test/pidnet_small_loveda_3b_AUG_CHANCE+AUG1+AUG2.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b AUG_CHANCE+AUG1+AUG3
python tools/train.py --cfg configs/loveDa/tests/3b_test/pidnet_small_loveda_3b_AUG_CHANCE+AUG1+AUG3.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b AUG_CHANCE+AUG1
python tools/train.py --cfg configs/loveDa/tests/3b_test/pidnet_small_loveda_3b_AUG_CHANCE+AUG1.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b AUG_CHANCE+AUG2+AUG3
python tools/train.py --cfg configs/loveDa/tests/3b_test/pidnet_small_loveda_3b_AUG_CHANCE+AUG2+AUG3.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b AUG_CHANCE+AUG2
python tools/train.py --cfg configs/loveDa/tests/3b_test/pidnet_small_loveda_3b_AUG_CHANCE+AUG2.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b AUG_CHANCE+AUG3
python tools/train.py --cfg configs/loveDa/tests/3b_test/pidnet_small_loveda_3b_AUG_CHANCE+AUG3.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b AUG_CHANCE
python tools/train.py --cfg configs/loveDa/tests/3b_test/pidnet_small_loveda_3b_AUG_CHANCE.yaml GPUS "[0]"

# PIDNet Small LoveDA Train AVD
python tools/train_ADV.py --cfg configs/loveDa/tests/4/pidnet_small_loveda_train_AVD.yaml GPUS "[0]"
# PIDNet Small LoveDA Train DACS
python tools/train_DACS.py --cfg configs/loveDa/tests/4/pidnet_small_loveda_train_DACS.yaml GPUS "[0]"

# PIDNet Small LoveDA 2b Fake Images
python tools/train.py --cfg configs/loveDa/tests/5/pidnet_small_loveda_2b_fake_images.yaml GPUS "[0]"
# PIDNet Small LoveDA 3b Fake Images
python tools/train.py --cfg configs/loveDa/tests/5/pidnet_small_loveda_3b_fake_images.yaml GPUS "[0]"
