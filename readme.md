# Implemention of AlexNet

> Under Development 

## Oh, Why the training does not work?

This project is an unofficial implemention of AlexNet, using C Program Language Without Any 3rd Library, according to the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky,et al.

## Original AlexNet Details

> divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and reduced three times prior to termination. We trained the network for roughly 90 cycles through the training set of 1.2 million images.

## Todo List

- [x]  **alexnet_backward**

- [ ]  **img2col conv**

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

## Reference

- [AlexNet, NIPS2012](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
