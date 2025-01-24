# TESTING SEMANTIC SEGMENTATION NETWORKS

## Real-time semantic segmentation network.

PIDNet on LovaDA-Urban (Train & Val) , 20 epochs 


| Numero | Optimizer | Loss  | Scheduler | Picture Size |  mIoU  | bestIoU | modified mIoU |
|--------|-----------|-------|-----------|--------------|--------|---------|---------------|
| 01_01  | Adam      | CE    | False     | 720x720      | 0.3165 | 0.3165  | 0.3617        |
| 02_00  | Adam      | CE    | False     | 1024x1024    | 0.3407 | 0.3417  | 0.3906        |
| 02_01  | Adam      | CE    | False     | 1024x1024    | 0.3254 | 0.3562  | 0.4071        |
| 03     | Adam      | CE    | False     | 720x720      | 0.3131 | 0.3261  | 0.3727        |
| 4      | Adam      | CE    | True      | 1024x1024    | 0.3245 | 0.3406  | 0.3893        |
| 5      | Adam      | OCE   | False     | 720x720      | 0.2895 | 0.2895  | 0.3318        |
| 7      | Adam      | OCE   | True      | 1024x1024    | 0.3421 | 0.3421  | 0.3910        |
| 8      | Adam      | OCE   | True      | 720x720      | 0.3381 | 0.3426  | 0.3915        |
| 9      | SDG       | OHEM  | False     | 720x720      | 0.3332 | 0.3385  | 0.3868        |
| 10     | SDG       | OHEM  | False     | 1024x1024    | 0.2588 | 0.2677  | 0.3059        |
| 11     | SDG       | CE    | False     | 720x720      | 0.2097 | 0.2301  | 0.2630        |
| 12     | SDG       | OHEM  | False     | 720x720      | 0.2896 | 0.2896  | 0.3310        |
| 13     | SDG       | DICE  | False     | 720x720      | 0.2387 | 0.2387  | 0.3442        |
| 14     | SDG       | FOCAL | False     | 720x720      | 0.1865 | 0.1958  | 0.2245        |
| 15     | SDG       | CE    | True      | 1024x1024    | 0.2943 | 0.3110  | 0.3554        |
| 16     | Adam      | DICE  | True      | 720x720      | 0.2289 |         | 0.3663        |
| 17     | Adam      | FOCAL | True      | 720x720      | 0.4113 |         | 0.4233        |


### Candidato
| 8      | Adam      | OCE   | True      | 720x720      | 0.3381 | 0.3426  | 0.3915        |

### 3A Result

| Numero | Optimizer | Loss  | Scheduler | Picture Size |  mIoU  | bestIoU | modified mIoU |
|--------|-----------|-------|-----------|--------------|--------|---------|---------------|
| 8      | Adam      | OCE   | True      | 720x720      | 0.1805 | 0.2009  | 0.2296        |


### 3b Results

| Numero | AUG_CHANCE | AUG1  | AUG2  | AUG3  |  mIoU  | bestIoU | modified mIoU |
|--------|------------|-------|-------|-------|--------|---------|---------------|
| 1      | TRUE       | False | False | False | 0.2656 | 0.2753  | 0.3146        |
| 2      | TRUE       | True  | False | False | 0.2523 | 0.2540  | 0.2903        |
| 3      | TRUE       | False | True  | False | 0.3089 |         | 0.3108        |
| 4      | TRUE       | True  | True  | False | 0.2672 | 0.2750  | 0.3143        |
| 5      | TRUE       | False | False | True  | 0.2568 | 0.2642  | 0.3020        |
| 6      | TRUE       | True  | False | True  | 0.2599 | 0.2632  | 0.3008        |
| 7      | TRUE       | False | True  | True  | 0.3133 |         | 0.3151        |
| 8      | TRUE       | True  | True  | True  | 0.2926 |         | 0.3014        |

### Best 3b result

| 7      | TRUE       | False | True  | True  | 0.3133 |         | 0.3151        |