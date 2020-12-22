//
// File:        activation_layer.c
// Description: Implementation of activation layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "activation_layer.h"


typedef struct nonlinear_args {
    nonlinear_op *op;
    short batch_id;
} nonlinear_args;

static void* pthread_relu_op_forward(void *argv)
{
    /**
     * pthread relu_op_forward
     * */
    nonlinear_args nargs;
    memcpy(&nargs, (nonlinear_args *)argv, sizeof(nonlinear_args));

    register float *input = nargs.op->input + nargs.batch_id * (nargs.op->units);
    register float *output =  nargs.op->output + nargs.batch_id * (nargs.op->units);
    for (register int i = 0; i < (nargs.op->units); i++)
    {
        if(input[i]>0)
        {
            output[i] = input[i];
        }else{
            output[i] = 0;
        }
    }
}

void relu_op_forward(nonlinear_op *op)
{
    nonlinear_args args[op->batchsize+1];
    pthread_t tid[op->batchsize+1];
    for(int p=0; p<op->batchsize; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        pthread_create(&tid[p], NULL, pthread_relu_op_forward, (void *)(&args[p]));
    }
    for(int p=0; p<op->batchsize; p++)
        pthread_join(tid[p], NULL);
}


void relu_op_backward(nonlinear_op *op)
{
    register float tmp;
    register float *input = op->input;
    for (register int i = 0; i < (op->units); i++)
    {
        tmp=0;
        for (int p = 0; p < op->batchsize; p++)
            tmp += (*(input++) > 0);
        tmp /= op->batchsize;

        op->d_input[i] = op->d_output[i] * tmp;
    }
}


void sigmoid_op_forward(nonlinear_op *op)
{
    for (int p=0; p< op->batchsize; p++)
    {
        for (int i = 0; i < (op->units); i++)
        {
            op->output[p*(op->units)+i] = 1.0 / ( 1.0 + exp( 0.0 - op->input[p*(op->units)+i] ) );
        }
    }
}


void sigmoid_op_backward(nonlinear_op *op)
{
    float tmp;
    for (int i = 0; i < (op->units); i++)
    {
        tmp=0;
        for (int p = 0; p < op->batchsize; p++)
            tmp += op->output[p*(op->units)+i] * (1 - op->output[p*(op->units)+i]);
        tmp /= op->batchsize;

        op->d_input[i] = op->d_output[i] * tmp;
    }
}


void softmax_op_forward(nonlinear_op *op)
{
    for(int p=0; p< op->batchsize; p++)
    {
        float esum=0;
        for(int i=0; i< op->units; i++)
            esum += exp( op->input[i + p * op->units]); 

        for(int i=0; i< op->units; i++)
            op->output[i + p * op->units] = exp( op->input[i + p * op->units]) / esum;       
    }
}


void softmax_op_backward(nonlinear_op *op)
{
    for(int i=0; i< op->units; i++)
    {
        for(int j=0; j< op->units; j++)
        {
            if(i==j){
                op->d_input[j] += op->d_output[j] * (1 - op->d_output[i]);
            }else{
                op->d_input[j] -= op->d_output[j] * op->d_output[i];
            }
        }
    }

    for(int i=0; i< op->units; i++)
    {
        op->d_input[i] /= op->units;
    }
}