# Implemention of AlexNet7

> Under Development 

This project is an unofficial implemention of AlexNet-7, using C Program Language Without Any 3rd Library, according to the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky,et al.

## Original AlexNet7 Details

- $\surd$ ~~Gaussian Initialization(mean=0,stddv=0.01)~~ XavierInitialization for all $w$s
- $\surd$ OneInitialization for all $b$s
- $\surd$ ReLU
- Dropout in the first two FC layers
- Data Augmentation: generating image translations & horizontal reflections
- ~~Local Response Normalization~~ Here I use Batch-Norm instead of LR-Norm
- $\surd$ Overlapping Pooling
- $\surd$ MomentumSGD
- $\surd$ CrossEntropy Loss
- CUDA / 2 GPU / Training

## Todo List
  
- **Dataset: loader**

- **Dataset: sampler**

- **Operation: Dropout regularization**

- **Full test**

- **Moudulr & Decoupling**
 
- **Trainer: logger**

- **Trainer: lr-scheduler** 
  
- **Trainer: weights save/load**

- **CUDA speed boosting**

- **Try & Compare on ImageNet**

- **circle loss**

- **Finally, an overall review on this project**

- $\surd$ **unit testing for all ops**

- $\surd$ **net_forward testing**

- $\surd$ **net_backward testing**

- $\surd$ **parameters initilization**

- $\surd$ **Metrics: Accuracy**

- $\surd$ **Metrics: Precision**

- $\surd$ **Metrics: Recall**

- $\surd$ **MomentumSGD**
