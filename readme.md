# Implemention of AlexNet7

> Under Development 

This project is an unofficial implemention of AlexNet-7, using C Program Language Without Any 3rd Library, according to the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky,et al.

## Original AlexNet7 Details

- GaussianInitialization(mean=0,stddv=0.01) for all $w$s
- OneInitialization for all $b$s
- ReLU
- Dropout in the first two FC layers
- Data Augmentation: generating image translations & horizontal reflections
- ~~Local Response Normalization~~ Here I use Batch-Norm instead of LR-Norm
- Overlapping Pooling
- MomentumSGD
- CrossEntropy Loss
- CUDA / 2 GPU / Training

## Todo List

- **Dropout**

- **MomentumSGD**
  
- **Dataset: loader**

- **Dataset: sampler**

- **Operation: Dropout regularization**

- **Metrics: Accuracy**

- **Metrics: Precision**

- **Metrics: Recall**

- **Full test**

- **Moudulr & Decoupling**
 
- **Trainer: logger**

- **Trainer: lr-scheduler** 
  
- **Trainer: weights save/load**

- **circle loss**

- **CUDA speed boosting**

- **Finally, an overall evaluation on this project**

- $\surd$ **unit testing for all ops**

- $\surd$ **net_forward testing**

- $\surd$ **net_backward testing**

- $\surd$ **parameters initilization**
