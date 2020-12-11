# Implemention of AlexNet7

> Under Development 

This project is an unofficial implemention of AlexNet-7, using C Program Language Without Any 3rd Library, according to the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky,et al.

## Original AlexNet7 Details

- ~~Gaussian Initialization(mean=0,stddv=0.01)~~ XavierInitialization for all $w$s
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

- [ ]  **Full test**

- [ ]  **Try & Compare on ImageNet**
 
- [ ]  **Trainer: logger**

- [ ]  **Moudulr & Decoupling**

- [ ]  **Trainer: lr-scheduler** 
  
- [ ]  **Trainer: weights save/load**

- [ ]  **CUDA speed boosting**

- [ ]  **circle loss**

- [ ]  **Finally, an overall review on this project**

- [x]  **unit testing for all ops**

- [x]  **net_forward testing**

- [x]  **net_backward testing**

- [x]  **parameters initilization**

- [x]  **Metrics: Accuracy**

- [x]  **Metrics: Precision**

- [x]  **Metrics: Recall**

- [x]  **MomentumSGD**

- [x]  **Operation: Dropout regularization**

- [x]  **Dataset: loader**

- [x]  **Dataset: sampler**
