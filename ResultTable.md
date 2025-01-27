# TESTING SEMANTIC SEGMENTATION NETWORKS


## Classic semantic segmentation network

DeepLab V2 on LoveDA-Urban (Train & Val) , 20 epochs
 
| Numero | Optimizer | Loss  | Scheduler | Picture Size |  mIoU  | bestIoU | modified mIoU |
|--------|-----------|-------|-----------|--------------|--------|---------|---------------|




## Real-time semantic segmentation network.

PIDNet on LoveDA-Urban (Train & Val) , 20 epochs 


| Numero | Optimizer | Loss  | Scheduler | Picture Size | mIoU          | Latency | FLOPs     | Parameters |
|--------|-----------|-------|-----------|--------------|---------------|---------|-----------|------------|
| 1     | Adam      | CE    | False     | 720x720      | 0.3617        | 2:45 hours | 1.10e+12  | 6.14e+07   |
| 2     | Adam      | CE    | False     | 1024x1024    | 0.3906        | 2:59 hours        | 1.10e+12  | 6.14e+07   |
| 3     | Adam      | CE    | True      | 720x720      | 0.3727        | 2:41 hours        | 1.10e+12  | 6.14e+07   |
| 4      | Adam      | CE    | True      | 1024x1024    | 0.3893        | 2:51 hours        | 1.10e+12  | 6.14e+07   |
| 5      | Adam      | OCE   | False     | 720x720      | 0.3318        | 2:42 hours        | 1.10e+12  | 6.14e+07   |
| 7      | Adam      | OCE   | True      | 1024x1024    | 0.3910        |         | 1.10e+12  | 6.14e+07   |
| 8      | Adam      | OCE   | True      | 720x720      | 0.3915        | 2:40 hours        | 1.10e+12  | 6.14e+07   |
| 9      | SDG       | OHEM  | False     | 720x720      | 0.3868        | 1:23 hours        | 1.10e+12  | 6.14e+07   |
| 10     | SDG       | OHEM  | False     | 1024x1024    | 0.3059        |         | 1.10e+12  | 6.14e+07   |
| 11     | SDG       | CE    | False     | 720x720      | 0.2630        |         | 1.10e+12  | 6.14e+07   |
| 12     | SDG       | OHEM  | False     | 720x720      | 0.3310        |         | 1.10e+12  | 6.14e+07   |
| 13     | SDG       | DICE  | False     | 720x720      | 0.3442        |         | 1.10e+12  | 6.14e+07   |
| 14     | SDG       | FOCAL | False     | 720x720      | 0.2245        |         | 1.10e+12  | 6.14e+07   |
| 15     | SDG       | CE    | True      | 1024x1024    | 0.3554        |         | 1.10e+12  | 6.14e+07   |
| 16     | Adam      | DICE  | True      | 720x720      | 0.3663        |         | 1.10e+12  | 6.14e+07   |
| 17     | Adam      | FOCAL | True      | 720x720      | 0.4233        |         | 1.10e+12  | 6.14e+07   |


### Candidato
| 8      | Adam      | OCE   | True      | 720x720      | 0.3381 | 0.3426  | 0.3915        |

### 3A Result

| Numero | Optimizer | Loss  | Scheduler | Picture Size |  mIoU  | bestIoU | modified mIoU |
|--------|-----------|-------|-----------|--------------|--------|---------|---------------|
| 8      | Adam      | OCE   | True      | 720x720      | 0.1805 | 0.2009  | 0.2296        |


### 3b Results

| Numero | AUG_CHANCE | AUG1  | AUG2  | AUG3  |  mIoU  | bestIoU | modified mIoU |
|--------|------------|-------|-------|-------|--------|---------|---------------|
| 1      | TRUE       | False | False | False | 0.2656 | 0.2753  | 0.2951      |
| 2      | TRUE       | True  | False | False | 0.3011 |         | 0.3042        |
| 3      | TRUE       | False | True  | False | 0.3089 |         | 0.3108        |
| 4      | TRUE       | True  | True  | False | 0.2672 | 0.2750  | 0.3143        |
| 5      | TRUE       | False | False | True  | 0.2568 | 0.2642  | 0.3020        |
| 6      | TRUE       | True  | False | True  | 0.2599 | 0.2632  | 0.3008        |
| 7      | TRUE       | False | True  | True  | 0.3133 |         | 0.3151        |
| 8      | TRUE       | True  | True  | True  | 0.2926 |         | 0.3014        |

### Best 3b result

| Numero | AUG_CHANCE | AUG1  | AUG2  | AUG3  |  mIoU  | bestIoU | modified mIoU |
|--------|------------|-------|-------|-------|--------|---------|---------------|
| 7      | TRUE       | False | True  | True  | 0.3133 |         | 0.3151        |