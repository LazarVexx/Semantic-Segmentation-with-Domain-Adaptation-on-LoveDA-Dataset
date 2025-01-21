# TESTING SEMANTIC SEGMENTATION NETWORKS

## Real-time semantic segmentation network.

PIDNet on LovaDA-Urban (Train & Val) , 20 epochs 

| Numero    | Optimizer | Loss | Scheduler  | Picture Size | mIoU   | bestIoU |
|--------   |-----------|------|----------- |--------------|------  |---------|
| 01_00     | Adam      | CE   | False      | 720x720      | 0.3131 | 0.3261  |
| 01_01     | Adam      | CE   | False      | 720x720      | 0.3165 | 0.3165  |
| 02_00     | Adam      | CE   | False      | 1024x1024    | 0.3407 | 0.3417  |
| 02_01     | Adam      | CE   | False      | 1024x1024    | 0.3254 | 0.3562  |
| 3         | Adam      | CE   | True       | 720x720      | 0,3261 | 0,3562  | Latitante
| 4         | Adam      | CE   | True       | 1024x1024    | 0.3245 | 0.3406  | 
| 5         | Adam      | OCE  | False      | 720x720      | 0.2895 | 0.2895  | 
| 6         | Adam      | OCE  | True       | 1024x1024    | 0.3445 | 0.3528  | Latitante
| 7         | Adam      | OCE  | True       | 1024x1024    | 0.3421 | 0.3421  |
| 8         | Adam      | OCE  | True       | 720x720      | 0.3381 | 0.3426  |
| 9         | SDG       | OHEM | False      | 720x720      | 0.3332 | 0.3385  |
| 10        | SDG       | OHEM | False      | 1024x1024    | 0.2588 | 0.2677  |
| 11        | SDG       | CE   | False      | 720x720      | 0.2097 | 0.2301  |
| 12        | SDG       | OHEM | False      | 720x720      | 0.2896 | 0.2896  |
| 13        | SDG       | DICE | False      | 720x720      | 0.2387 | 0.2387  |
| 14        | SDG       | FOCAL| False      | 720x720      | 0.1865 | 0.1958  |
| 15        | SDG       | CE   | True       | 1024x1024    | 0.2943 | 0.3110  |

