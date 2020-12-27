//
// File:        maxpooling_layer.c
// Description: Implementation of max pooling layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "maxpooling_layer.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))


typedef struct mp_args{
    max_pooling_op *op;
    short batch_id;
} mp_args;

static void* pthread_mp_op_forward(void *argv)
{
    mp_args mp;
    memcpy(&mp, (mp_args *)argv, sizeof(mp_args));
    register float *input  = mp.op->input + mp.batch_id * mp.op->in_units;
    register float *output = mp.op->output + mp.batch_id * mp.op->out_units;
    int channels  = mp.op->channels;
    int strides   = mp.op->stride;
    int pool_size = mp.op->kernel_size;

    register int o_x, o_y;
    register int input_offset;
    register int output_offset;
    register int iwih = mp.op->in_w * mp.op->in_h;
    register int owoh = mp.op->out_w * mp.op->out_h;

    for (register int c = 0; c < channels; c++)
    {
        
        o_y=0;
        for (int j = 0; j < mp.op->in_h-strides+1; j += strides)
        {
            o_x=0;
            for (int i = 0; i < mp.op->in_w-strides+1; i += strides)
            {
                /**
                 * inputs[i ~ i+pool_size][j ~ j+pool_size]
                 * outputs[o_x][o_j]
                 * */
                input_offset = i + j*(mp.op->in_w) + c*iwih;
                register float pixel = input[input_offset];
                for (register int fx=i; fx<MIN(i+pool_size,mp.op->in_w); fx++)
                {
                    for (register int fy=j; fy<MIN(j+pool_size,mp.op->in_h); fy++)
                    {                        
                        pixel = MAX(pixel, input[fx*mp.op->in_h+fy]);
                    }
                }
                output_offset = o_x + o_y*(mp.op->out_w) + c*owoh;
                output[output_offset] = pixel;
                o_x++;
            }
            o_y++;
        }
    }
}

void max_pooling_op_forward(max_pooling_op *op)
{
    mp_args args[op->batchsize+1];
    pthread_t tid[op->batchsize+1];
    for (int p = 0; p < op->batchsize; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        pthread_create(&tid[p], NULL, pthread_mp_op_forward, (void *)(&args[p]));
    }
    
    for (int p = 0; p < op->batchsize; p++)
        pthread_join(tid[p], NULL);
}

void max_pooling_op_backward(max_pooling_op *op)
{
    int channels  = op->channels;
    int pool_size = op->kernel_size;
    int in_w  = op->in_w; 
    int in_h  = op->in_h;
    int out_w = op->out_w; 
    int out_h = op->out_h;
    register int iwih = in_w*in_h;
    register int owoh = out_w*out_h;

    int in_x, in_y;
    float max_value, cur_value;
    int x, y;
    register int in_shift, out_shift;
    for (int c = 0; c < channels; c++)
    {
        for (int i = 0; i < op->out_w; i++)
        {
            for (int j = 0; j < op->out_h; j++)
            {
                for (int p = 0; p < op->batchsize; p++)
                {
                    //
                    // output[p][c][i][j]
                    //
                    x = i*pool_size;    
                    y = j*pool_size;
                    max_value = -1111111;
                    while ( x < MIN((i + 1) * pool_size, in_w) )
                    {
                        while ( y < MIN((j + 1) * pool_size, in_h) )
                        {
                            cur_value = op->input[p*channels*iwih + c*iwih + y*in_w + x];
                            if (cur_value > max_value)
                            {
                                max_value = cur_value;
                                in_x = x;
                                in_y = y;
                            }
                            y++;
                        }
                        x++;
                    }
                    
                    in_shift  = c*iwih + in_y*in_w + in_x;
                    out_shift = c*owoh + j*out_w + i;
                    op->d_input[in_shift] += op->d_output[out_shift] / op->batchsize;
                }
            }
        }
    }
}
