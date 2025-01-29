# TESTING SEMANTIC SEGMENTATION NETWORKS


## Classic semantic segmentation network

DeepLab V2 on LoveDA-Urban (Train & Val) , 20 epochs
 
| Numero | Optimizer | Loss  | Scheduler | Picture Size |  mIoU  | bestIoU | lr  | Latency (s) | FLOPs      | Params    |
|--------|-----------|-------|-----------|--------------|--------|---------|-----|-------------|------------|-----------|
|1       | Adam      | CE    | True      | 720x720      | 0.2514 | 0.2734  | 0.001| 0.005340    | 1.10e+12   | 6.14e+07  |
|2       | Adam      | DICE  | True      | 720x720      | 0.1323 | 0.1274  | 0.001| 0.005236    | 1.10e+12   | 6.14e+07  |
|3       | Adam      | FOCAL | True      | 720x720      | 0.2151 | 0.2559  | 0.001| 0.005734    | 1.10e+12   | 6.14e+07  |
|4       | Adam      | OCE   | True      | 720x720      | 0.2408 | 0.2687  | 0.001| 0.005383    | 1.10e+12   | 6.14e+07  |
|5       | SGD       | CE    | True      | 720x720      | 0.3318 | 0.3364  | 0.01 | 0.005062    | 1.10e+12   | 6.14e+07  |
|6       | SGD       | DICE  | True      | 720x720      | 0.1822 | 0.3112  | 0.01 | 0.005163    | 1.10e+12   | 6.14e+07  |
|7       | SGD       | FOCAL | True      | 720x720      | 0.2466 | 0.2761  | 0.01 | 0.005105    | 1.10e+12   | 6.14e+07  |
|8       | SGD       | OCE   | True      | 720x720      | 0.3267 | 0.3473  | 0.01 | 0.004744    | 1.10e+12   | 6.14e+07  |
|9       | SGD       | DICE  | True      | 720x720      | 0.3501 | 0.3610  | 0.001| 0.004934    | 1.10e+12   | 6.14e+07  |
|10      | SGD       | CE    | True      | 720x720      | 0.3428 | 0.3526  | 0.001| 0.005232    | 1.10e+12   | 6.14e+07  |
|11      | SGD       | OCE   | True      | 720x720      | 0.3412 | 0.3422  | 0.001| 0.005422    | 1.10e+12   | 6.14e+07  |



### Candidate
|9|SGD|DICE|True|720x720|0.3501|0.3610|0.001|




## Real-time semantic segmentation network.

PIDNet on LoveDA-Urban (Train & Val) , 20 epochs 


| Numero | Optimizer | Loss  | Scheduler | Picture Size | mIoU   | Latency (sec) | FLOPs     | Parameters |
|--------|-----------|-------|-----------|--------------|--------|---------------|-----------|------------|
| 1      | Adam      | CE    | False     | 720x720      | 0.3617 | 0.029         | 1.10e+12  | 6.14e+07   |
| 2      | Adam      | CE    | False     | 1024x1024    | 0.3906 | 0.027         | 1.10e+12  | 6.14e+07   |
| 3      | Adam      | CE    | True      | 720x720      | 0.3727 | 0.029         | 1.10e+12  | 6.14e+07   |
| 4      | Adam      | CE    | True      | 1024x1024    | 0.3893 | 0.027         | 1.10e+12  | 6.14e+07   |
| 5      | Adam      | OHEM  | False     | 720x720      | 0.3318 | 0.034         | 1.10e+12  | 6.14e+07   |
| 6      | Adam      | OHEM  | True      | 1024x1024    | 0.4275 | 0.033         | 1.10e+12  | 6.14e+07   |
| 7      | Adam      | OHEM  | True      | 720x720      | 0.4368 | 0.030         | 1.10e+12  | 6.14e+07   |
| 8      | Adam      | DICE  | True      | 720x720      | 0.3663 | 0.033         | 1.10e+12  | 6.14e+07   |
| 9      | Adam      | FOCAL | True      | 720x720      | 0.4233 | 0.033         | 1.10e+12  | 6.14e+07   |
| 10     | SDG       | OHEM  | False     | 720x720      | 0.3868 | 0.035         | 1.10e+12  | 6.14e+07   |
| 11     | SDG       | OHEM  | False     | 1024x1024    | 0.3059 | 0.031         | 1.10e+12  | 6.14e+07   |
| 12     | SDG       | CE    | False     | 720x720      | 0.2630 | 0.029         | 1.10e+12  | 6.14e+07   |
| 13     | SDG       | OHEM  | True      | 720x720      | 0.3657 | 0.033         | 1.10e+12  | 6.14e+07   |
| 14     | SDG       | DICE  | False     | 720x720      | 0.3442 | 0.033         | 1.10e+12  | 6.14e+07   |
| 15     | SDG       | FOCAL | False     | 720x720      | 0.2245 | 0.033         | 1.10e+12  | 6.14e+07   |
| 16     | SDG       | CE    | True      | 1024x1024    | 0.3554 | 0.027         | 1.10e+12  | 6.14e+07   |



### Candidato
| 7      | Adam      | OCE   | True      | 720x720      | 0.3381 | 0.3426  | 0.3915        |

### 3a Result

| Numero | Optimizer | Loss  | Scheduler | Picture Size | mIoU | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|--------|-----------|-------|-----------|--------------|---------------|----------|--------|--------|--------|---------|-----------|----------|
| 8      | Adam      | OCE   | True      | 720x720      | 0.2296        | 0.4158   | 0.2176 | 0.1666 | 0.3349 | 0.0590  | 0.1415    | 0.2716   |



### 3b Results

| Numero | AUG_CHANCE | AUG1  | AUG2  | AUG3  | mIoU | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|--------|------------|-------|-------|-------|---------------|----------|--------|--------|--------|---------|-----------|----------|
| Default| False      | False | False | False | 0.2296        | 0.4158   | 0.2176 | 0.1666 | 0.3349 | 0.0590  | 0.1415    | 0.2716   |
| 1      | TRUE       | False | False | False | 0.2951        | 0.5217   | 0.3381 | 0.3098 | 0.3188 | 0.0673  | 0.0839    | 0.4262   |
| 2      | TRUE       | True  | False | False | 0.3042        | 0.5255   | 0.3789 | 0.3074 | 0.4121 | 0.0377  | 0.0265    | 0.4417   |
| 3      | TRUE       | False | True  | False | 0.3108        | 0.4900   | 0.3403 | 0.3097 | 0.4075 | 0.0582  | 0.1526    | 0.4170   |
| 4      | TRUE       | True  | True  | False | 0.3143        | 0.4766   | 0.3495 | 0.3304 | 0.3810 | 0.0682  | 0.1779    | 0.4165   |
| 5      | TRUE       | False | False | True  | 0.3020        | 0.5257   | 0.3998 | 0.2933 | 0.3413 | 0.0708  | 0.0574    | 0.4257   |
| 6      | TRUE       | True  | False | True  | 0.3008        | 0.5102   | 0.3952 | 0.3130 | 0.3587 | 0.0457  | 0.0505    | 0.4324   |
| 7      | TRUE       | False | True  | True  | 0.3509        | 0.5306   | 0.2683 | 0.3407 | 0.4896 | 0.1072  | 0.2865    | 0.4332   |
| 8      | TRUE       | True  | True  | True  | 0.3014        | 0.4877   | 0.3868 | 0.3008 | 0.3700 | 0.0586  | 0.1589    | 0.3472   |


### Best 3b result

| Numero | AUG_CHANCE | AUG1  | AUG2  | AUG3  | modified mIoU | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|--------|------------|-------|-------|-------|---------------|----------|--------|--------|--------|---------|-----------|----------|
| 7      | TRUE       | False | True  | True  | 0.3509        | 0.5306   | 0.2683 | 0.3407 | 0.4896 | 0.1072  | 0.2865    | 0.4332   |

## 4th STEP DOMAIN ADAPTATION

### 4a Adversarial approach

| Modello               | mIoU   | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|-----------------------|--------|----------|--------|--------|--------|---------|-----------|----------|
| PIDNet ADV            | 0.2770 | 0.5145   | 0.2651 | 0.2679 | 0.3808 | 0.1306  | 0.0585    | 0.3217   |

### 4b Image-to-image approach

| Modello               | mIoU   | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|-----------------------|--------|----------|--------|--------|--------|---------|-----------|----------|
| PIDNet  DACS          | 0.2918 | 0.5454   | 0.3345 | 0.2913 | 0.4343 | 0.1016  | 0.2310    | 0.3959  |



### CycleGAN: 2b Best on Fake Images, 20 epochs

| Modello      | Training Set          | Target Set     | Test Set             | mIoU   | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|--------------|-----------------------|----------------|----------------------|--------|----------|--------|--------|--------|---------|-----------|----------|
| PIDNet       | CycleGAN LoveDa-Urban | LoveDa-Rural   | CycleGAN LoveDa-Urban| 0.4035 | 0.3508   | 0.4845 | 0.5442 | 0.6434 | 0.0919  | 0.3659    | 0.3440   |



### CycleGAN: 3b Best on Fake Images, 20 epochs

| Modello       | Training Set          | Target Set     | Test Set             | mIoU   | Building | Road   | Water  | Barren | Forest  | Grassland | Farmland |
|---------------|-----------------------|----------------|----------------------|--------|----------|--------|--------|--------|---------|-----------|----------|
| PIDNet        | CycleGAN LoveDa-Urban | LoveDa-Rural   | LoveDa-Rural         | 0.2880 | 0.5127   | 0.1962 | 0.3027 | 0.4716 | 0.0625  | 0.0353    | 0.4349   |



## Result PEM

| Codice | Modello               | Training Set          | Test Set          | mIoU    | fwIoU   | mACC    | pACC    |
|--------|-----------------------|-----------------------|-------------------|---------|---------|---------|---------|
| 01     | PEM-URBAN             | LoveDa-Urban          | LoveDa-Urban      | 64.4429 | 60.3522 | 75.1795 | 74.5604 |
| 02     | PEM-RURAL             | LoveDa-Rural          | LoveDa-Rural      | 44.5885 | 56.3823 | 54.6906 | 71.6951 |
| 03     | PEM-CycleGAN-RURAL    | CycleGAN LoveDa-Urban | LoveDa-Rural      | 46.8514 | 54.9652 | 62.2863 | 68.8383 |
