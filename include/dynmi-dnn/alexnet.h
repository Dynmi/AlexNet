//
// File:        alexnet.h
// Description: alexnet.h
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "convolution_layer.h"
#include "maxpooling_layer.h"
#include "activation_layer.h"
#include "fc_layer.h"
#include "batchnorm_layer.h"
#include "dropout_layer.h"

#define SHOW_PREDCITION_DETAIL
//#define SHOW_METRIC_EVALUTE
//#define SHOW_OP_TIME


//
//  Definition of model shape
//
#define IN_CHANNELS 3
#define C1_CHANNELS 64
#define C2_CHANNELS 192
#define C3_CHANNELS 384
#define C4_CHANNELS 256
#define C5_CHANNELS 256

#define C1_KERNEL_L 11
#define C2_KERNEL_L 5
#define C3_KERNEL_L 3
#define C4_KERNEL_L 3
#define C5_KERNEL_L 3

#define C1_STRIDES 4
#define C2_STRIDES 1
#define C3_STRIDES 1
#define C4_STRIDES 1
#define C5_STRIDES 1

#define C1_PADDING 2
#define C2_PADDING 2
#define C3_PADDING 1
#define C4_PADDING 1
#define C5_PADDING 1

#define FEATURE0_L 224
#define FEATURE1_L 55
#define POOLING1_L 27
#define FEATURE2_L 27
#define POOLING2_L 13
#define FEATURE3_L 13
#define FEATURE4_L 13
#define FEATURE5_L 13
#define POOLING5_L 6

#define FC6_LAYER   4096
#define FC7_LAYER   4096
#define OUT_LAYER   1000
#define DROPOUT_PROB  0.4


typedef struct network {

    float *input;
    float *output;
    short batchsize;
    
    conv_op conv1;
    batch_norm_op bn1;
    nonlinear_op relu1;

    max_pooling_op mp1;

    conv_op conv2;
    batch_norm_op bn2;
    nonlinear_op relu2;

    max_pooling_op mp2;

    conv_op conv3;
    batch_norm_op bn3;
    nonlinear_op relu3;

    conv_op conv4;
    batch_norm_op bn4;
    nonlinear_op relu4;

    conv_op conv5;
    batch_norm_op bn5;
    nonlinear_op relu5;

    max_pooling_op mp5;

    fc_op fc1;
    nonlinear_op relu6;

    fc_op fc2;
    nonlinear_op relu7;

    fc_op fc3;
} alexnet;


//
//  Definiation of metric type
//
#define METRIC_ACCURACY  0
#define METRIC_PRECISION 1      // macro-precision
#define METRIC_RECALL    2      // macro-recall
#define METRIC_F1SCORE   3
#define METRIC_ROC       4

void metrics(float *ret, int *preds, int *labels, 
                int classes, int TotalNum, int type);
int argmax(float *arr, int n);


void malloc_alexnet(alexnet *net);
void free_alexnet(alexnet *net);

void set_alexnet(alexnet *net, short batchsize, char *weights_path);

void forward_alexnet(alexnet *net);
void backward_alexnet(alexnet *net, int *batch_Y);

void alexnet_train(alexnet *net, int epochs);
void alexnet_inference(alexnet *net, char *filename);
