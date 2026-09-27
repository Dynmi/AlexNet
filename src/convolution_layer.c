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
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size; //number of pixels in a channel of a kernel
    int ikk  = op->in_channels * kk; //total number of pixels in a kernel
    int pad  = op->padding;

    // st_x,y are the top-left corner of the receptive field in the (possibly
    // negative, zero-padded) input coordinate system.
    int out_y = 0;
    for (int st_y = -pad; out_y < op->out_h; st_y += op->stride, out_y++)
    {
        int out_x = 0;
        for (int st_x = -pad; out_x < op->out_w; st_x += op->stride, out_x++)
        {
            // row-major: out_y slow, out_x fast -- matches load_image's image
            // layout and max_pooling_layer.c's read/write convention.
            int patch_idx = out_y * op->out_w + out_x;

            for (int in_c = 0; in_c < op->in_channels; in_c++)
            {
                int x_col_offset = patch_idx * ikk + in_c * kk;

                for (int j = 0; j < op->kernel_size; j++)
                {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        int src_x = st_x + i;
                        int src_y = st_y + j;
                        if (src_x >= 0 && src_x < op->in_w && src_y >= 0 && src_y < op->in_h)
                        {
                            int input_offset = src_x + src_y * op->in_w + in_c * iwih;
                            col[x_col_offset] = img[input_offset];
                        }
                        else
                        {
                            col[x_col_offset] = 0.0f;
                        }
                        x_col_offset++;
                    }
                }
            }
        }
    }
} //so we destroy data locality in this algorithm in order to have data locality in GEMM


static void* pthread_conv_op_forward(void *argv)
{
    
    /**
     * pthread conv_op_forward
     * 
     * argv is a pointer in struct type conv_args
     * */
    conv_args cp;
    memcpy(&cp, (conv_args *)argv, sizeof(conv_args));
    int col_size_per_image = (cp.op->in_channels * cp.op->kernel_size * cp.op->kernel_size) * (cp.op->out_w * cp.op->out_h);
    float *x_col = cp.op->input_col + cp.batch_id * col_size_per_image;
    // float *x_col    = cp.op->input_col + cp.batch_id * cp.op->in_units; i changed it...
    float *t_input  = cp.op->input + cp.batch_id * cp.op->in_units; //take the image that is for the image that your thread uses
    float *t_output = cp.op->output + cp.batch_id * cp.op->out_units; // store the output of the convolution in the right, channel
    int ikk  = cp.op->in_channels * cp.op->kernel_size * cp.op->kernel_size;  //weight dimension j
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


    return NULL;
}

/*

typedef struct conv_args{
    conv_op *op;
    short batch_id;
    short st_tunits;
    short ed_tunits;
} conv_args;
*/


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
    conv_args args[op->batchsize+1]; //each image in the batch has its own struct
    pthread_t tid[op->batchsize+1];
    for (int p = 0; p < op->batchsize; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        pthread_create(&tid[p], NULL, pthread_conv_op_forward, (void *)(&args[p])); //each thread takes a struct on an image
    } //tosa threads osa kai ta images sto batch
    
    for (int p = 0; p < op->batchsize; p++)
        pthread_join(tid[p], NULL);

}


static void col2img(const float *col, float *img, const conv_op *op)
{
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size;
    int ikk  = op->in_channels * kk;
    int pad  = op->padding;

    // Must mirror img2col's (st_x,st_y,patch_idx) mapping exactly -- this is
    // its inverse (scatter-add instead of gather).
    int out_y = 0;
    for (int st_y = -pad; out_y < op->out_h; st_y += op->stride, out_y++)
    {
        int out_x = 0;
        for (int st_x = -pad; out_x < op->out_w; st_x += op->stride, out_x++)
        {
            int patch_idx = out_y * op->out_w + out_x;

            for (int in_c = 0; in_c < op->in_channels; in_c++)
            {
                int x_col_offset = patch_idx * ikk + in_c * kk;

                for (int j = 0; j < op->kernel_size; j++)
                {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        int src_x = st_x + i;
                        int src_y = st_y + j;
                        if (src_x >= 0 && src_x < op->in_w && src_y >= 0 && src_y < op->in_h)
                        {
                            int input_offset = src_x + src_y * op->in_w + in_c * iwih;
                            img[input_offset] += col[x_col_offset];
                        }
                        x_col_offset++;
                    }
                }
            }
        }
    }
}


static void* pthread_conv_op_backward(void *argv)
{
    /**
     * pthread conv_op_backward
     * 
     * dLoss/dw= dLoss/dy * dy/dw
     * 
     * dy/dw=x
     * 
     * <=> dLoss/dw=dLoss/dy * x <=> d_weights = d_output * input_col
     * 
     * d_output is now [batchsize, oc*owoh] (per-sample gradients)
     * */
    conv_args args;
    memcpy(&args, (conv_args *)argv, sizeof(conv_args));

    int oc   = args.op->out_channels,
        ikk  = args.op->in_channels * args.op->kernel_size * args.op->kernel_size, 
        owoh = args.op->out_w * args.op->out_h; //pixels of ΟΝΕ channel of output fmap

    // calculate delta_weights using per-sample d_output
    short internal     = args.ed_tunits - args.st_tunits; //workload of the thread in this function
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
        // use this sample's d_output
        matrix_multiply(args.op->d_output + p * oc * owoh, t_input_col, t_d_weights, oc, owoh, internal);

        // t_d_weights[oc, internal] = dW^T. Must store as dW[ikk, oc] to match weights layout.
        for (int j = 0; j < oc; j++)
        {
            for (int i = 0; i < internal; i++)
                args.op->d_weights[(args.st_tunits + i) * oc + j] += t_d_weights[j*internal + i] / args.op->batchsize; 
        }
    }
    free(t_d_weights);
    free(t_input_col);

    if (args.st_tunits == 0 )
    {
        // calculate delta_bias averaged across batch
        for (int i = 0; i < args.op->out_channels; i++)
        {
            register float tmp = 0.0f;
            for (int p = 0; p < args.op->batchsize; p++)
                for (int s = i*owoh; s < (i+1)*owoh; s++)
                    tmp += args.op->d_output[p * oc * owoh + s];
            args.op->d_bias[i] = tmp / args.op->batchsize;
        }

        // calculate delta_input per sample
        float *weights_T = (float *)malloc(ikk * oc * sizeof(float));
        memcpy(weights_T, args.op->weights, ikk * oc * sizeof(float));
        matrix_transpose(weights_T, ikk, oc);

        float *d_out_copy = (float *)malloc(oc * owoh * sizeof(float));
        float *d_x_col = (float *)malloc(ikk * owoh * sizeof(float));

        for (int p = 0; p < args.op->batchsize; p++)
        {
            // copy this sample's d_output and transpose it
            memcpy(d_out_copy, args.op->d_output + p * oc * owoh, oc * owoh * sizeof(float));
            // d_out_copy is [oc, owoh], transpose to [owoh, oc]
            matrix_transpose(d_out_copy, oc, owoh);
            memset(d_x_col, 0, ikk * owoh * sizeof(float));
            // d_x_col[owoh, ikk] = d_output[owoh, oc] @ W^T[oc, ikk]
            matrix_multiply(d_out_copy, weights_T, d_x_col, owoh, oc, ikk);
            col2img(d_x_col, args.op->d_input + p * args.op->in_units, args.op);
        }

        free(d_out_copy);
        free(d_x_col);
        free(weights_T);
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
    short tnum = 8; // number of threads
    int ikk = op->in_channels * op->kernel_size * op->kernel_size; //pixels of patch/kernel
    if (ikk < tnum) //if threads less than pixels of a kernel/filter
    {
        conv_args args;
        args.op = op;
        args.st_tunits = 0;
        args.ed_tunits = ikk;
        pthread_conv_op_backward((void *)(&args));
    }else {
        conv_args args[tnum+1];
        pthread_t tid[tnum+1];
        short internal = ceil(1.0 * ikk / tnum); //workload per thread
        short actual_threads = 0;

        for (int p = 0; p < tnum; p++)
        {
            args[p].op = op;
            args[p].st_tunits = p*internal;
            args[p].ed_tunits = MIN(args[p].st_tunits+internal, ikk);
            if (args[p].st_tunits >= ikk) break;
            actual_threads++; // how many threads created, in order to join them
            pthread_create(&tid[p], NULL, pthread_conv_op_backward, (void *)(&args[p]));
        }

        for (int p = 0; p < actual_threads; p++)
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
