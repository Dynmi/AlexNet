//
// File:        batchnorm_layer.c
// Description: Implementation of batch normalization layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "batchnorm_layer.h"

void batch_norm_op_forward(batch_norm_op *op)
{
    register float *input  = op->input;
    register float *output = op->output;

    op->avg    = (float *)calloc(op->units, sizeof(float));
    op->var    = (float *)calloc(op->units, sizeof(float));
    op->x_norm = (float *)malloc(op->batchsize * (op->units) * sizeof(float));

    register int i, p;

    // calculate mean for each unit along batch axis
    register int offset = 0;
    for (p = 0; p < op->batchsize; p++)
    {
        for (i = 0; i < op->units; i++)
            op->avg[i] += input[offset++];
    }
    for (i = 0; i < op->units; i++)
        op->avg[i] /= op->batchsize;

    // calculate variance for each unit along batch axis
    offset = 0;
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
    offset = 0;
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
    int   units       = op->units;
    float *x_norm_avg = (float *)calloc(units, sizeof(float));

    // calculate delta_gamma
    register int offset = 0;
    for (int p = 0; p < op->batchsize; p++)
    {
        for (int i = 0; i < units; i++)
            op->d_gamma[i] += op->x_norm[offset++] * op->d_output[i];
    }
    for (int i = 0; i < units; i++)
        op->d_gamma[i] /= op->batchsize;

    // calculate delta_beta
    for (int i = 0; i < units; i++)
        op->d_beta[i] += op->d_output[i];

    // calculate average of normlized x along batch axis
    offset = 0;
    for (int p = 0; p < op->batchsize; p++)
    {
        for (int i = 0; i < units; i++)
            x_norm_avg[i] += op->x_norm[offset++];
    }
    for (int i = 0; i < units; i++)
        x_norm_avg[i] /= op->batchsize;

    // calculate delta_input
    for (int i = 0; i < units; i++)
        op->d_input[i] = 0 - op->gamma[i] * op->d_output[i] * x_norm_avg[i] / sqrt(op->var[i]+EPSILON); 

    free(x_norm_avg);
    free(op->var);
    free(op->x_norm);
}


void calloc_batchnorm_weights(batch_norm_op *op)
{
    op->gamma = (float *)calloc(op->units, sizeof(float));
    op->beta  = (float *)calloc(op->units, sizeof(float));
}

void free_batchnorm_weights(batch_norm_op *op)
{
    free(op->gamma);
    free(op->beta);
}

void calloc_batchnorm_dweights(batch_norm_op *op)
{
    op->d_gamma = (float *)calloc(op->units, sizeof(float));
    op->d_beta  = (float *)calloc(op->units, sizeof(float));
}

void free_batchnorm_dweights(batch_norm_op *op)
{
    free(op->d_gamma);
    free(op->d_beta);
}

void save_batchnorm_weights(batch_norm_op *op, FILE *fp)
{
    fwrite(op->gamma, sizeof(float), op->units, fp);
    fwrite(op->beta,  sizeof(float), op->units, fp);
}

void load_batchnorm_weights(batch_norm_op *op, FILE *fp)
{
    fread(op->gamma, sizeof(float), op->units, fp);
    fread(op->beta,  sizeof(float), op->units, fp);
}