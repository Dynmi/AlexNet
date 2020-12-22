//
// File:        batchnorm_layer.c
// Description: Implementation of batch normalization layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include "batchnorm_layer.h"

void batch_norm_op_forward(batch_norm_op *op)
{
    register float *input = op->input;
    register float *output = op->output;

    op->avg = (float *)calloc(op->units, sizeof(float));
    op->var = (float *)calloc(op->units, sizeof(float));
    op->x_norm = (float *)malloc(sizeof(float) * op->batchsize * (op->units));

    register int i, p;
    // calculate mean & variance for each unit along batch axis
    register int offset=0;
    for (p = 0; p < op->batchsize; p++)
    {
        for (i = 0; i < op->units; i++)
            op->avg[i] += input[offset++];
    }
    for (i = 0; i < op->units; i++)
        op->avg[i] /= op->batchsize;

    offset=0;
    for (p = 0; p < op->batchsize; p++)
    {
        for (i = 0; i < op->units; i++)
        {
            register float tmp = input[offset++] - op->avg[i];
            op->var[i] += tmp*tmp;
        }
    }
    for (i = 0; i < op->units; i++)
        op->var[i] /= op->batchsize;

    register float *gamma = op->gamma;
    register float *beta = op->beta;
    offset=0;
    for (p = 0; p < op->batchsize; p++)
    {
        for (i = 0; i < op->units; i++)
        {
            // calculate normalized x 
            op->x_norm[offset] = (input[offset] - op->avg[i]) / sqrt(op->var[i] + EPSILON); 
            // scale & shift
            output[offset] = gamma[i] * op->x_norm[offset] + beta[i];
            offset++;
        }
    }

    free(op->avg);
}


void batch_norm_op_backward(batch_norm_op *op)
{
    int units = op->units;
    float *x_norm_avg = (float *)calloc(units, sizeof(float));

    // calculate d_gamma
    register int offset=0;
    for(int p=0; p<op->batchsize; p++)
    {
        for(int i=0; i<units; i++)
            op->d_gamma[i] += op->x_norm[offset++] * op->d_output[i];
    }
    for(int i=0; i<units; i++)
        op->d_gamma[i] /= op->batchsize;

    // calculate d_beta
    for(int i=0; i<units; i++)
        op->d_beta[i] += op->d_output[i];

    // calculate x_norm_average
    offset=0;
    for(int p=0; p<op->batchsize; p++)
    {
        for(int i=0; i<units; i++)
            x_norm_avg[i] += op->x_norm[offset++];
    }

    for(int i=0; i<units; i++)
        x_norm_avg[i] /= op->batchsize;

    for(int i=0; i<units; i++)
        op->d_input[i] = 0 - op->gamma[i] * op->d_output[i] * x_norm_avg[i] / sqrt(op->var[i]+EPSILON); 

    free(x_norm_avg);
    free(op->var);
    free(op->x_norm);
}
