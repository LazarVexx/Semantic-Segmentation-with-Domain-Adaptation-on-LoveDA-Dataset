### INTRODUCTION
Semantic segmentation is a fundamental task in computer vision that involves classifying each pixel in an image into a predefined category, thereby enabling detailed scene understanding. Recent advancements in deep learning have significantly improved the accuracy and efficiency of semantic segmentation methods. For instance, DeepLab employs atrous convolutions and fully connected Conditional Random Fields (CRFs) to capture multi-scale context and refine segmentation boundaries \cite{chen2017deeplab}. BiSeNet introduces a bilateral network architecture to balance spatial detail and semantic context, achieving real-time performance \cite{yu2018bisenet}. Meanwhile, PIDNet draws inspiration from PID controllers to develop a lightweight network for high-quality real-time segmentation \cite{feng2021pidnet}. Domain adaptation techniques further enhance semantic segmentation by addressing challenges posed by domain shifts, as seen in methods like DACS, which utilizes cross-domain mixed sampling \cite{tranheden2021dacs}, and LoveDA, a dataset specifically designed for domain adaptation in remote sensing \cite{wang2021loveda}. These advancements highlight the diverse strategies employed to tackle the complexities of semantic segmentation across different applications and domains.

### RELATED WORK
## DeepLab

## PIDNet

PIDNet is a real-time semantic segmentation network inspired by the principles of Proportional-Integral-Derivative (PID) controllers, which are widely used in control systems to achieve precise and stable performance \cite{feng2021pidnet}. By integrating ideas from PID control theory, PIDNet introduces a unique architecture that balances low-latency processing with high-quality segmentation results. The network comprises three branches—P (proportional), I (integral), and D (derivative)—that are designed to capture complementary information: the P-branch focuses on spatial detail, the I-branch accumulates global context, and the D-branch enhances boundary precision. This innovative design enables PIDNet to achieve state-of-the-art performance in real-time segmentation tasks, particularly in scenarios that demand both accuracy and efficiency, such as autonomous driving and robotics. Furthermore, its lightweight structure ensures applicability in resource-constrained environments without significant performance degradation, demonstrating its practical utility in a wide range of applications.

## LoveDA



### METHODOLOGY

## DeepLab realizzazione

## PIDNet realizzazione

## Domain Shift and Augmentation theory

## Adversarial

## DACS

## CycleGAN

## PEM


### EXPERIMENTS and RESULTS

## DeepLab Tests and table

## PIDNet Tests and table

## Data Augmentation Tests and table

## Adversarial Tests and table

## DACS Tests and table

## CycleGAN Tests and table

## PEM Tests and table



### CONCLUSION
