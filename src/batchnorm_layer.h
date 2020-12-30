//
// File:        batchnorm_layer.h
// Description: interface of batch normalization layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>

#define EPSILON 0.00001

typedef struct batch_norm_op {
    float *input; float *d_input;
    float *output; float *d_output;
    float *gamma; float *d_gamma;
    float *beta; float *d_beta;

    int units;
    short batchsize;
    
    float *x_norm;
    float *avg;
    float *var;
} batch_norm_op;

void batch_norm_op_forward(batch_norm_op *op);
void batch_norm_op_backward(batch_norm_op *op);

inline void calloc_batchnorm_weights(batch_norm_op *op);
inline void free_batchnorm_weights(batch_norm_op *op);

inline void calloc_batchnorm_dweights(batch_norm_op *op);
inline void free_batchnorm_dweights(batch_norm_op *op);

inline void load_batchnorm_weights(batch_norm_op *op, FILE *fp);
inline void save_batchnorm_weights(batch_norm_op *op, FILE *fp);
