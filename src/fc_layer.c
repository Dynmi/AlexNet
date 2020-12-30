//
// File:        fc_layer.c
// Description: Implementation of full connected layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include "fc_layer.h"
#include "matrix.h"
#define MIN(a,b) (((a) < (b)) ? (a) : (b))


typedef struct fc_args{
    fc_op *op;
    short batch_id;
    short st_tunits;
    short ed_tunits;
} fc_args;

static void* pthread_fc_op_forward(void *argv)
{
    /**
     * pthread fc_op_forward
     * */
    fc_args args;
    memcpy(&args, (fc_args *)argv, sizeof(fc_args));
    short internal   = args.ed_tunits - args.st_tunits;
    float *t_weights = (float *)malloc(internal * (args.op->in_units) * sizeof(float));
    for (int j = 0; j < args.op->in_units; j++)
    {   
        memcpy((void *)(t_weights+j*internal), 
                (void *)(args.op->weights+j*(args.op->out_units)+args.st_tunits), 
                    sizeof(float)*internal);
    }

    float *t_output = (float *)calloc(internal * (args.op->batchsize), sizeof(float));
    matrix_multiply(args.op->input, t_weights, t_output,  args.op->batchsize,  args.op->in_units, internal);

    for (int j = 0; j < args.op->batchsize; j++)
    {
        register int o_offset  = j * internal;
        register int oo_offset = j * (args.op->out_units) + args.st_tunits;

        for (int i = 0; i < internal; i++, o_offset++, oo_offset++)
            args.op->output[oo_offset] = t_output[o_offset] + args.op->bias[args.st_tunits+i]; 
    }
    free(t_output);
    free(t_weights);
}

void fc_op_forward(fc_op *op)
{
    short tnum = 12; // number of threads
    if (op->out_units < tnum)
    {
        fc_args args;
        args.op = op;
        args.st_tunits = 0;
        args.ed_tunits = op->out_units;
        pthread_fc_op_forward((void *)(&args));
    }else {
        fc_args args[tnum+1];
        pthread_t tid[tnum+1];
        short internal = ceil(1.0 * op->out_units / tnum);
    
        for (int p = 0; p < tnum; p++)
        {
            args[p].op = op;
            args[p].st_tunits = p*internal;
            args[p].ed_tunits = MIN(args[p].st_tunits+internal, op->out_units);            
            pthread_create(&tid[p], NULL, pthread_fc_op_forward, (void *)(&args[p]));
        }

        for (int p = 0; p < tnum; p++)
            pthread_join(tid[p], NULL);
    }

}

static void* pthread_fc_op_backward(void *argv)
{
    /**
     * pthread fc_op_backward
     * */
    fc_args args;
    memcpy(&args, (fc_args *)argv, sizeof(fc_args));

    register int w_offset=0;
    if (args.st_tunits == 0)
    {
        // calculate delta_bias and delta_input
        for (register int j = 0; j < args.op->out_units; j++)
        {
            register float d_o = args.op->d_output[j];
            for (register int i = 0; i < args.op->in_units; i++, w_offset++)
            {
                args.op->d_input[i] += args.op->weights[w_offset] * d_o;
            }
            args.op->d_bias[j] = d_o;
        }
    }
 
    register float *w_deltas  = args.op->d_weights;
    for (int i = args.st_tunits; i < args.ed_tunits; i++)
    {
        register float oe = args.op->d_output[i];
        if (oe < 0.0008)
            continue;

        register float *input = args.op->input;
        for (int p = 0; p < args.op->batchsize; p++)
        {
            w_offset = i * (args.op->in_units);
            register int w_o_bound = w_offset + args.op->in_units;
            while (w_offset < w_o_bound)
            {
                w_deltas[w_offset++] += oe * (*(input++)) / args.op->batchsize;
            }
        }
    }
}

void fc_op_backward(fc_op *op)
{
    short tnum = 12; // number of threads
    if (op->out_units < tnum) {
        fc_args args;
        args.op = op;
        args.st_tunits = 0;
        args.ed_tunits = op->out_units;
        pthread_fc_op_backward((void *)(&args));
    }else {
        fc_args args[tnum+1];
        pthread_t tid[tnum+1];
        short internal = ceil(1.0 * op->out_units / tnum);

        for (int p = 0; p < tnum; p++)
        {
            args[p].op = op;
            args[p].st_tunits = p*internal;
            args[p].ed_tunits = MIN(args[p].st_tunits+internal, op->out_units);            
            pthread_create(&tid[p], NULL, pthread_fc_op_backward, (void *)(&args[p]));
        }

        for (int p = 0; p < tnum; p++)
            pthread_join(tid[p], NULL);
    }

}


void calloc_fc_weights(fc_op *op)
{
    op->weights = (float *)calloc(op->in_units * op->out_units, sizeof(float));
    op->bias    = (float *)calloc(op->out_units, sizeof(float));
}

void free_fc_weights(fc_op *op)
{
    free(op->weights);
    free(op->bias);
}

void calloc_fc_dweights(fc_op *op)
{
    op->d_weights = (float *)calloc(op->in_units * op->out_units, sizeof(float));
    op->d_bias    = (float *)calloc(op->out_units, sizeof(float));
}

void free_fc_dweights(fc_op *op)
{
    free(op->d_weights);
    free(op->d_bias);
}

void save_fc_weights(fc_op *op, FILE *fp)
{
    fwrite(op->weights, sizeof(float), op->in_units * op->out_units, fp);
    fwrite(op->bias,    sizeof(float), op->out_units, fp);
}

void load_fc_weights(fc_op *op, FILE *fp)
{
    fread(op->weights, sizeof(float), op->in_units * op->out_units, fp);
    fread(op->bias,    sizeof(float), op->out_units, fp);
}
