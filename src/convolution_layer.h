//
// File:        convolution_layer.h
// Description: interface of convolution layer
// Author:      Haris Wang
//
#include <stdlib.h>

typedef struct conv_op {
    float *input;   float *d_input;
    float *output;  float *d_output;
    float *weights; float *d_weights;
    float *bias;    float *d_bias;
    float *input_col;

    int in_channels, out_channels;
    int kernel_size; int padding; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    short batchsize;
} conv_op;

             
void conv_op_forward(conv_op *op);
void conv_op_backward(conv_op *op);

inline void calloc_conv_weights(conv_op *op);
inline void free_conv_weights(conv_op *op);

inline void calloc_conv_dweights(conv_op *op);
inline void free_conv_dweights(conv_op *op);

inline void load_conv_weights(conv_op *op, FILE *fp);
inline void save_conv_weights(conv_op *op, FILE *fp);
