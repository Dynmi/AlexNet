//
// File:        fc_layer.h
// Description: interface of full connected layer
// Author:      Haris Wang
//
#include <stdlib.h>

typedef struct fc_op {
    float *input;   float *d_input;
    float *output;  float *d_output;
    float *weights; float *d_weights;
    float *bias;    float *d_bias;
    int in_units, out_units;

    short batchsize;
} fc_op;


void fc_op_forward(fc_op *op);
void fc_op_backward(fc_op *op);

inline void calloc_fc_weights(fc_op *op);
inline void free_fc_weights(fc_op *op);

inline void calloc_fc_dweights(fc_op *op);
inline void free_fc_dweights(fc_op *op);

inline void load_fc_weights(fc_op *op, FILE *fp);
inline void save_fc_weights(fc_op *op, FILE *fp);
