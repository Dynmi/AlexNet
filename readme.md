# Implementation of AlexNet

> Under Development 

- ! ! ! It needs an evaluation on ImageNet

This project is an unofficial implementation of AlexNet, using C Program Language Without Any 3rd Library, according to the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky,et al.

**Only support CPU now**

## Features

- **Effective matrix multiply**, w.r.t L1/L2 cache
- **img2col** implementation of convolutional layer
- **Multi-thread CPU** Apply MT in operator's forward computation and backward computation to boost speed 
- **Efficient Memory Usage & Minimal Memory Occupation** Memory is allocated on demand for efficient memory usage. If an array isn't necessary for computations afterward, it's memory will be freed instantly to minimize memory occupation.
- **Modular layer** Define each layer seperately

## Architecture

```
----------------------------------------------------------------
            Layer                Output Shape         Param #
================================================================
            Conv2d-1           [N, 64, 55, 55]          23,296
              ReLU-2           [N, 64, 55, 55]               0
         MaxPool2d-3           [N, 64, 27, 27]               0
            Conv2d-4          [N, 192, 27, 27]         307,392
              ReLU-5          [N, 192, 27, 27]               0
         MaxPool2d-6          [N, 192, 13, 13]               0
            Conv2d-7          [N, 384, 13, 13]         663,936
              ReLU-8          [N, 384, 13, 13]               0
            Conv2d-9          [N, 256, 13, 13]         884,992
             ReLU-10          [N, 256, 13, 13]               0
           Conv2d-11          [N, 256, 13, 13]         590,080
             ReLU-12          [N, 256, 13, 13]               0
        MaxPool2d-13            [N, 256, 6, 6]               0
          Dropout-14                 [N, 9216]               0
           Linear-15                 [N, 4096]      37,752,832
             ReLU-16                 [N, 4096]               0
          Dropout-17                 [N, 4096]               0
           Linear-18                 [N, 4096]      16,781,312
             ReLU-19                 [N, 4096]               0
           Linear-20                 [N, 1000]       4,097,000
================================================================
("N" stands for "batch size")
Total params: 61,100,840
Trainable params: 61,100,840
Non-trainable params: 0
----------------------------------------------------------------
```

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

- [ ] **Data: get_next_batch()**

- [ ]  **Try & Compare on ImageNet**

- [ ]  **CUDA speed boosting**

- [ ]  **Report for Speed Benchmark**


## Difference from Torch/Tensorflow/Caffe/Darknet

- ? Maybe... will be faster than all of them on CPU ? later I will put a benchmark here.

## Original AlexNet Details

> divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. 
> The learning rate was initialized at 0.01 and reduced three times prior to termination. 
> We trained the network for roughly 90 cycles through the training set of 1.2 million images.

## Reference

- [AlexNet, NIPS2012](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

- [Darknet](https://github.com/AlexeyAB/darknet)

- [PyTorch](https://github.com/pytorch/pytorch)
