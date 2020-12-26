# Implementation of AlexNet

> Under Development 

- It needs an evaluation on imagenet!!!!!!!!!!!!!!!!!!

This project is an unofficial implementation of AlexNet, using C Program Language Without Any 3rd Library, according to the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky,et al.

**Only support CPU now**.

## Features
- **Effective matrix multiply**, w.r.t L1/L2 cache
- **img2col** implementation of convolutional layer
- **Multi-thread CPU** Apply MT in operator's forward computation and backward computation to boost speed 
- **Modular layer** Define each layer seperately

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

## Usage (Only for Linux now)

### Install
```
git clone git@github.com:Dynmi/AlexNet.git
cd AlexNet
make clean && make all
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
./alexnet train -batchsize <batch-size> -epochs <epochs> -load_pretrained <weights-path> -save <weights-path>
```

For example:
```
./alexnet train -batchsize 8 -epochs 10 -load_pretrained ./alexnet_pretrained.weights -save ./temp.weights 
```

### Inference
```
$./alexnet inference -input <image-path> -load <weights-path>
```

For example:
```
./alexnet inference -input ./0001.jpeg -load ./alexnet_pretrained.weights
```

## Todo List

- [ ]  **Try & Compare on ImageNet**

- [ ]  **CUDA speed boosting**

- [ ]  **circle loss**

## Original AlexNet Details

> divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. 
> The learning rate was initialized at 0.01 and reduced three times prior to termination. 
> We trained the network for roughly 90 cycles through the training set of 1.2 million images.

## Reference

- [AlexNet, NIPS2012](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

- [Darknet](https://github.com/AlexeyAB/darknet)
