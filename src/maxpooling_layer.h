//
// File:        maxpooling_layer.h
// Description: interface of max pooling layer
// Author:      Haris Wang
//
#include <stdlib.h>


typedef struct max_pooling_op {
    float *input;  float *d_input;
    float *output; float *d_output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    short batchsize;
} max_pooling_op;

void max_pooling_op_forward(max_pooling_op *op);
void max_pooling_op_backward(max_pooling_op *op);
