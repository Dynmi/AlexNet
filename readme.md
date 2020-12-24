# Implementation of AlexNet

> Under Development 

- It needs an evaluation on imagenet!!!!!!!!!!!!!!!!!!

This project is an unofficial implementation of AlexNet, using C Program Language Without Any 3rd Library, according to the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky,et al.

**Only support CPU now**.

## Architecture

- Input
- Conv2D
- BatchNorm
- Relu
- MaxPooling
- Conv2D
- BatchNorm
- Relu
- MaxPooing
- Conv2D
- BatchNorm
- Relu
- Conv2D
- BatchNorm
- Relu
- Conv2D
- BatchNorm
- Relu
- MaxPooling
- Dropout
- FullConnected
- Relu
- Dropout
- FullConnected
- Relu
- FullConnected
- Output

## Original AlexNet Details

> divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and reduced three times prior to termination. We trained the network for roughly 90 cycles through the training set of 1.2 million images.

## Usage

### Install
```
git clone git@github.com:Dynmi/AlexNet.git
cd AlexNet
make all
```

### Train

1. Put the training images in the directory ```./images/``` as followed:
( only ```.jpeg``` and ```.png$``` image supported now)
```
├── images
│   ├── 0
│   │   ├── 1.jpeg
│   │   ├── 2.jpeg
│   │   ├── 3.jpeg
│   │   ├── ...
│   │   └── x.jpeg
│   ├── 1
│   │   ├── ...
│   ├── 2
│   │   ├── ...
│   ├── 3
│   │   ├── ...
|   ...
│   └── 999
└──────

```

2. 
```
./alexnet train -batchsize <batchsize> -epochs <epochs>
```

### Inference (not done yet)

```
./alexnet inference <image-path>
```

## Todo List

- [ ]  **Trainer: weights save/load**

- [ ]  **Try & Compare on ImageNet**

- [ ]  **CUDA speed boosting**

- [ ]  **circle loss**

## Features
- Effective matrix multiply, w.r.t L1 cache and L2 cache
- **img2col** implementation of convolution layer
- **Multi-thread CPU** speed boosting for net forward and net backward
- **Modular layer** implementation

## Reference

- [AlexNet, NIPS2012](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

- [Darknet](https://github.com/AlexeyAB/darknet)
