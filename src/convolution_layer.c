//
// File:        convolution_layer.c
// Description: Implementation of convolution layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "convolution_layer.h"
#include "matrix.h"
#define MIN(a,b) (((a) < (b)) ? (a) : (b))


typedef struct conv_args{
    conv_op *op;
    short batch_id;
    short st_tunits;
    short ed_tunits;
} conv_args;

static void img2col(const float *img, float *col, const conv_op *op)
{
    /**
     * Output
     *      col[ikk][owoh]
     * */

    //
    // Todo: simplify the code
    //
    register int input_offset;
    register int iwih = op->in_w*op->in_h;
    register int kk   = op->kernel_size* op->kernel_size;
    register int ikk  = op->in_channels * kk;
    register float *input = img;
    register float *x_col = col;
    for (register unsigned short in_c = 0; in_c < op->in_channels; in_c++)
    {
        register int x_col_offset = in_c * kk;
        for (register int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride)
        {
            for (register int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, x_col_offset += ikk)
            {
                for (register unsigned short j = 0; j < op->kernel_size; j++)
                {
                    for (register unsigned short i = 0; i < op->kernel_size; i++, x_col_offset++)
                    {
                        if (!(st_x+i <op->in_w) | !(st_y+j <op->in_h))
                        {
                            x_col[x_col_offset] = 0;
                            continue;
                        }

                        input_offset = (st_x+i) + (st_y+j) * op->in_w + in_c * iwih;
                        x_col[x_col_offset] = input[input_offset];
                    }
                }
            }
        }
        ikk += kk;
    }
}

static void col2img(const float *col, float *img, const conv_op *op)
{
    //
    // Todo: simplify the code
    //
    register int input_offset;
    register int iwih = op->in_w * op->in_h;
    int kk  = op->kernel_size * op->kernel_size;
    int ikk = op->in_channels * kk;

    register int st_x=0;
    for (register unsigned short out_x = 0; out_x < op->out_w; out_x++)
    {
        register int st_y=0;
        for (register unsigned short out_y = 0; out_y < op->out_h; out_y++)
        {
            for (register unsigned short in_c = 0; in_c < op->in_channels; in_c++)
            {    
                register int x_col_offset = in_c * kk + out_x*out_y*ikk;
                for (register unsigned short j = 0; j < op->kernel_size; j++)
                {
                    for (register unsigned short i = 0; i < op->kernel_size; i++, x_col_offset++)
                    {
                        if (!(st_x+i < op->in_w) | !(st_y+j < op->in_h))
                            continue;
                        
                        input_offset = (st_x+i) + (st_y+j) * op->in_w + in_c * iwih;
                        img[input_offset] = col[x_col_offset];
                    }
                }
            }
            st_y += op->stride;
        }
        st_x += op->stride;
    }
}

static void* pthread_conv_op_forward(void *argv)
{
    /**
     * pthread conv_op_forward
     * */
    conv_args cp;
    memcpy(&cp, (conv_args *)argv, sizeof(conv_args));

    float *x_col    = cp.op->input_col + cp.batch_id * cp.op->in_units;
    float *t_input  = cp.op->input + cp.batch_id * cp.op->in_units;
    float *t_output = cp.op->output + cp.batch_id * cp.op->out_units;
    int ikk  = cp.op->in_channels * cp.op->kernel_size * cp.op->kernel_size;
    int owoh = cp.op->out_w * cp.op->out_h;
    // 
    // >>>>>>>shape<<<<<<<
    //  
    // t_input    [ic,ih,iw]
    // x_col      [owoh,ikk]
    // weights    [ikk,oc]
    // t_output   [oc,oh,ow]
    // >>>>>>>>>>>>>>>>>>>
    //
    img2col(t_input, x_col, cp.op);
    matrix_multiply(x_col, cp.op->weights, t_output, owoh, ikk, cp.op->out_channels); //output[owoh,oc]
    matrix_transpose(t_output, owoh, cp.op->out_channels); //output[oc,owoh]

    register int o_offset=0;
    for (int i = 0; i < cp.op->out_channels; i++)
    {
        register float tmp = cp.op->bias[i];
        while (o_offset < (i+1)*owoh)
        {
            t_output[o_offset++] += tmp;
        }
    }

}

void conv_op_forward(conv_op *op)
{
    /**
     * conv2d forward
     * 
     * Input:
     *      op->input
     *      op->weights
     *      op->bias
     * Output:
     *      op->output
     * */
    op->input_col = (float *)calloc((op->batchsize)*(op->in_channels * op->kernel_size* op->kernel_size)*(op->out_w * op->out_h), sizeof(float));
    conv_args args[op->batchsize+1];
    pthread_t tid[op->batchsize+1];
    for (int p = 0; p < op->batchsize; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        pthread_create(&tid[p], NULL, pthread_conv_op_forward, (void *)(&args[p]));
    }
    
    for (int p = 0; p < op->batchsize; p++)
        pthread_join(tid[p], NULL);

}


static void* pthread_conv_op_backward(void *argv)
{
    /**
     * pthread conv_op_backward
     * */
    conv_args args;
    memcpy(&args, (conv_args *)argv, sizeof(conv_args));

    int oc   = args.op->out_channels,
        ikk  = args.op->in_channels * args.op->kernel_size * args.op->kernel_size, 
        owoh = args.op->out_w * args.op->out_h;

    // calculate delta_weights
    short internal     = args.ed_tunits - args.st_tunits;
    float *t_input_col = (float *)malloc( owoh * internal * sizeof(float));
    float *t_d_weights = (float *)malloc( oc * internal * sizeof(float));
    for (int p = 0; p < args.op->batchsize; p++)
    {
        for (int j = 0; j < owoh; j++)
        {
            memcpy((void *)(t_input_col+j*internal), 
                    (void *)(args.op->input_col+p*owoh*ikk + j*ikk + args.st_tunits),
                        sizeof(float)*internal);
        }
        memset(t_d_weights, 0,  oc * internal * sizeof(float));
        matrix_multiply(args.op->d_output, t_input_col, t_d_weights, oc, owoh, internal);

        for (int j = 0; j < oc; j++)
        {
            register int o_offset = j*internal;
            register int oo_offset = j*ikk + args.st_tunits;
            for (int i = 0; i < internal; i++)
                args.op->d_weights[oo_offset++] += t_d_weights[o_offset++] / args.op->batchsize; 
        }
    }
    free(t_d_weights);
    free(t_input_col);

    if (args.st_tunits == 0 )
    {
        // calculte delta_input and delta_bias
        for (int i = 0; i < args.op->out_channels; i++)
        {
            register int tmp=0;
            for (int p = i*owoh; p < (i+1)*owoh; p++)
                tmp += args.op->d_output[p];
            args.op->d_bias[i] = tmp;
        }

        float *d_x_col = (float *)calloc(ikk*owoh, sizeof(float));
        matrix_transpose(args.op->d_output, owoh, args.op->out_channels);
        matrix_multiply(args.op->d_output, args.op->weights, d_x_col, owoh, args.op->out_channels, ikk);
        col2img(d_x_col, args.op->d_input, args.op);
        free(d_x_col);
    }
}

void conv_op_backward(conv_op *op)
{
    /**
     * conv2d backward
     * 
     * Input:
     *      op->d_output
     * Output:
     *      op->d_weights
     *      op->d_bias
     *      op->d_input
     * */
    short tnum = 12; // number of threads
    if (op->in_channels * op->kernel_size * op->kernel_size < tnum)
    {
        conv_args args;
        args.op = op;
        args.st_tunits = 0;
        args.ed_tunits = op->in_channels * op->kernel_size * op->kernel_size;
        pthread_conv_op_backward((void *)(&args));
    }else {
        conv_args args[tnum+1];
        pthread_t tid[tnum+1];
        short internal = ceil(1.0 * op->in_channels * op->kernel_size * op->kernel_size / tnum);

        for (int p = 0; p < tnum; p++)
        {
            args[p].op = op;
            args[p].st_tunits = p*internal;
            args[p].ed_tunits = MIN(args[p].st_tunits+internal, op->in_channels * op->kernel_size * op->kernel_size);            
            pthread_create(&tid[p], NULL, pthread_conv_op_backward, (void *)(&args[p]));
        }

        for (int p = 0; p < tnum; p++)
            pthread_join(tid[p], NULL);
    }
    free(op->input_col);

}

void calloc_conv_weights(conv_op *op)
{
    op->weights = (float *)calloc(op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, sizeof(float));
    op->bias    = (float *)calloc(op->out_channels, sizeof(float));
}

void free_conv_weights(conv_op *op)
{
    free(op->weights);
    free(op->bias);
}

void calloc_conv_dweights(conv_op *op)
{
    op->d_weights = (float *)calloc(op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, sizeof(float));
    op->d_bias    = (float *)calloc(op->out_channels, sizeof(float));
}

void free_conv_dweights(conv_op *op)
{
    free(op->d_weights);
    free(op->d_bias);
}

void save_conv_weights(conv_op *op, FILE *fp)
{
    fwrite(op->weights, sizeof(float), op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, fp);
    fwrite(op->bias,    sizeof(float), op->out_channels, fp);
}


void load_conv_weights(conv_op *op, FILE *fp)
{
    fread(op->weights, sizeof(float), op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, fp);
    fread(op->bias,    sizeof(float), op->out_channels, fp);
}
