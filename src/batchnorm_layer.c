//
// File:        batchnorm_layer.c
// Description: Implementation of spatial (per-channel) batch normalization
// Author:      Haris Wang
// Spatial batch normalization rewrite: Stavros Mitropoulos
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "batchnorm_layer.h"


/*

typedef struct batch_norm_op {
    float *input; float *d_input;
    float *output; float *d_output;
    float *gamma; float *d_gamma; --> per channel
    float *beta; float *d_beta; --> per channel

    int units;
    int channels;
    int spatial_size;
    short batchsize;
    
    float *x_norm;
    float *avg;
    float *var;
} batch_norm_op;
 
*/



void batch_norm_op_forward(batch_norm_op *op)
{
    register float *input  = op->input;
    register float *output = op->output;

    op->avg    = (float *)calloc(op->units, sizeof(float));
    op->var    = (float *)calloc(op->units, sizeof(float));
    op->x_norm = (float *)malloc(op->batchsize * (op->units) * sizeof(float));

    register int i, p, c;

    // calculate mean for each unit along batch axis
    // [image_0: unit_0, unit_1, ..., unit_N, image_1: unit_0, ..., image_B: ..., unit_N]
    //batch_idx * (C*H*W) + c*(H*W) + h*W + w

    register int offset = 0;

    for (p=0; p< op->batchsize; p++) {
        for (c=0; c < op->channels; c++) {
            for (i=0; i < op-> spatial_size; i++) {
                op->avg[c] += input[offset++];
            }
        }
    }
    offset=0;
    register float factor = 1.0f / (op->batchsize * op->spatial_size);
    register float diff = 0;
    for (c = 0; c < op->channels; c++)
        op->avg[c] *= factor;

    for (p=0; p< op->batchsize; p++) {
        for (c=0; c < op->channels; c++) {
            for (i=0; i < op-> spatial_size; i++) {
                diff = input[offset++] - op->avg[c];
                op->var[c] += diff * diff;
            }
        }
    }

    for (c = 0; c < op->channels; c++)
        op->var[c] *= factor;

    register float *gamma = op->gamma;
    register float *beta = op->beta;


    offset = 0;
    register float inv_std;

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < op->channels; c++) {
            inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            for (int s = 0; s < op->spatial_size; s++) {
                op->x_norm[offset] = (input[offset] - op->avg[c]) * inv_std;

                output[offset] = op->gamma[c] * op->x_norm[offset]  + op->beta[c];
                offset++;
            }
        }
    }

    // free(op->avg);
    //where to free in inference? memory leak
}


void batch_norm_op_backward(batch_norm_op *op)
{
    int channels = op->channels;
    int spatial_size = op->spatial_size;
    int total_elements = op->batchsize * channels * spatial_size;

    // calculate delta_gamma: d_gamma[c] = (1/B) * sum_{n,s} x_norm_{n,c,s} * d_output_{n,c,s}
    for (int c = 0; c < channels; c++)
        op->d_gamma[c] = 0.0f;

    register int offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                op->d_gamma[c] += op->x_norm[offset] * op->d_output[offset];
                offset++;
            }
        }
    }
    for (int c = 0; c < channels; c++)
        op->d_gamma[c] /= op->batchsize;

    // calculate delta_beta: d_beta[c] = (1/B) * sum_{n,s} d_output_{n,c,s}
    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                op->d_beta[c] += op->d_output[offset++];
            }
        }
    }
    for (int c = 0; c < channels; c++)
        op->d_beta[c] /= op->batchsize;

    // calculate d_input (per-sample)
    float *dxnorm = (float*)malloc(total_elements * sizeof(float));
    float *S1 = (float*)calloc(channels, sizeof(float));
    float *S2 = (float*)calloc(channels, sizeof(float));

    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                dxnorm[offset] = op->d_output[offset] * op->gamma[c];
                offset++;
            }
        }
    }

    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                float dxn = dxnorm[offset];
                float xn = op->x_norm[offset];
                S1[c] += dxn;
                S2[c] += dxn * xn;
                offset++;
            }
        }
    }

    float M = (float)(op->batchsize * spatial_size);
    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            float inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            float s1_c = S1[c];
            float s2_c = S2[c];
            for (int s = 0; s < spatial_size; s++) {
                float dxn = dxnorm[offset];
                float xn = op->x_norm[offset];
                float term = (s1_c + xn * s2_c) / M;
                op->d_input[offset] = inv_std * (dxn - term);
                offset++;
            }
        }
    }

    free(dxnorm);
    free(S1);
    free(S2);
    free(op->var);
    free(op->x_norm);
    free(op->avg);
}


void calloc_batchnorm_weights(batch_norm_op *op)
{
    op->gamma = (float *)calloc(op->channels, sizeof(float));
    op->beta  = (float *)calloc(op->channels, sizeof(float));
}

void free_batchnorm_weights(batch_norm_op *op)
{
    free(op->gamma);
    free(op->beta);
}

void calloc_batchnorm_dweights(batch_norm_op *op)
{
    op->d_gamma = (float *)calloc(op->channels, sizeof(float));
    op->d_beta  = (float *)calloc(op->channels, sizeof(float));
}

void free_batchnorm_dweights(batch_norm_op *op)
{
    free(op->d_gamma);
    free(op->d_beta);
}

void save_batchnorm_weights(batch_norm_op *op, FILE *fp)
{
    fwrite(op->gamma, sizeof(float), op->channels, fp);
    fwrite(op->beta,  sizeof(float), op->channels, fp);
}

void load_batchnorm_weights(batch_norm_op *op, FILE *fp)
{
    fread(op->gamma, sizeof(float), op->channels, fp);
    fread(op->beta,  sizeof(float), op->channels, fp);
}