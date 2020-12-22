#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "hyperparams.h"

//#define SHOW_PREDCITION_DETAIL
#define SHOW_METRIC_EVALUTE
#define SHOW_OP_TIME




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
#define OUT_LAYER   10



//
//  Declaration of operations
//
typedef struct conv_op {
    float *input; float *d_input;
    float *output; float *d_output;
    float *weights; float *d_weights;
    float *bias; float *d_bias;
    float *input_col;

    int in_channels, out_channels;
    int kernel_size; int padding; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;
} conv_op;

typedef struct max_pooling_op {
    float *input; float *d_input;
    float *output; float *d_output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;
} max_pooling_op;

typedef struct fc_op {
    float *input; float *d_input;
    float *output; float *d_output;
    float *weights; float *d_weights;
    float *bias; float *d_bias;
    int in_units, out_units;
} fc_op;

typedef struct batch_norm_op {
    float *input; float *d_input;
    float *output; float *d_output;
    float *gamma; float *d_gamma;
    float *beta; float *d_beta;

    int units;

    float *x_norm;
    float *avg;
    float *var;
} batch_norm_op;

typedef struct nonlinear_op {
    float *input; float *d_input;
    float *output; float *d_output;
    int units;
} nonlinear_op;

typedef struct network {

    float *input;
    float *output;

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
    nonlinear_op sfx;
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


void alexnet_forward(alexnet *net);
void alexnet_backward(alexnet *net, int *batch_Y);

void alexnet_malloc_params(alexnet *net);
void alexnet_free_params(alexnet *net);

void alexnet_param_init(alexnet *net);

void alexnet_train(alexnet *net, int epochs);

void alexnet_inference(alexnet *net);
