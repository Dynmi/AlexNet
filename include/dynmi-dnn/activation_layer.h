//
// File:        activation_layer.h
// Description: interface of activation layer
// Author:      Haris Wang
//
#include <stdlib.h>


typedef struct nonlinear_op {
    float *input; float *d_input;
    float *output; float *d_output;
    int units;

    short batchsize;
} nonlinear_op;

void relu_op_forward(nonlinear_op *op);
void relu_op_backward(nonlinear_op *op);

void sigmoid_op_forward(nonlinear_op *op);
void sigmoid_op_backward(nonlinear_op *op);

void softmax_op_forward(nonlinear_op *op);
void softmax_op_backward(nonlinear_op *op);