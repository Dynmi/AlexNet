//
// File:        train.c
// Description: Implementation of functions related to training
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "alexnet.h"
#include "data.h"
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define LEARNING_RATE 0.001

static struct timespec start, finish; 
static float duration;

static void cross_entropy_loss(float *delta_preds, const float *preds, const int *labels, int units, int BATCH_SIZE)
{
    /**
     * Cross Entropy backward
     * 
     * Input:
     *      preds       [BATCH_SIZE, units]
     *      labels      [BATCH_SIZE]
     * Output:
     *      delta_preds [units]
     * */
    float ce_loss = 0;
    for (int p = 0; p < BATCH_SIZE; p++)
    {
        register float esum = 0;
        for (int i = 0; i < units; i++)
            esum += exp(preds[i+p*units]);

        ce_loss += 0 - log(exp(preds[labels[p]+p*units]) / esum);

        for (int i = 0; i < units; i++)
        {
            // preds[i+p*units]
            if (labels[p] == i) {
                delta_preds[i] += exp(preds[i+p*units]) / esum - 1;
            }else {
                delta_preds[i] += exp(preds[i+p*units]) / esum;
            } 
        }
    }
    ce_loss /= BATCH_SIZE;
    printf("cross entropy loss on batch data is %f \n", ce_loss);
    return;

    // whether to use it ?
    for (int i = 0; i < units; i++)
        delta_preds[i] /= BATCH_SIZE;
}


static float v_conv1_weights[C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L];
static float v_conv2_weights[C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L];
static float v_conv3_weights[C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L];
static float v_conv4_weights[C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L];
static float v_conv5_weights[C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L];
static float v_fc1_weights[C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L];
static float v_fc2_weights[FC6_LAYER*FC7_LAYER];
static float v_fc3_weights[FC7_LAYER*OUT_LAYER];

static float v_conv1_bias[C1_CHANNELS];
static float v_conv2_bias[C2_CHANNELS];
static float v_conv3_bias[C3_CHANNELS];
static float v_conv4_bias[C4_CHANNELS];
static float v_conv5_bias[C5_CHANNELS];
static float v_fc1_bias[FC6_LAYER];
static float v_fc2_bias[FC7_LAYER];
static float v_fc3_bias[OUT_LAYER];

static float v_bn1_gamma[C1_CHANNELS*FEATURE1_L*FEATURE1_L];
static float v_bn2_gamma[C2_CHANNELS*FEATURE2_L*FEATURE2_L];
static float v_bn3_gamma[C3_CHANNELS*FEATURE3_L*FEATURE3_L];
static float v_bn4_gamma[C4_CHANNELS*FEATURE4_L*FEATURE4_L];
static float v_bn5_gamma[C5_CHANNELS*FEATURE5_L*FEATURE5_L];
static float v_bn1_beta[C1_CHANNELS*FEATURE1_L*FEATURE1_L];
static float v_bn2_beta[C2_CHANNELS*FEATURE2_L*FEATURE2_L];
static float v_bn3_beta[C3_CHANNELS*FEATURE3_L*FEATURE3_L];
static float v_bn4_beta[C4_CHANNELS*FEATURE4_L*FEATURE4_L];
static float v_bn5_beta[C5_CHANNELS*FEATURE5_L*FEATURE5_L];


static inline void CLIP(float *x, float down, float up)
{
    *x = MIN(up, MAX(down, *x));
}

static void momentum_sgd(float *w, float *v_w, float *d_w, int units)
{
    /**
     * momentum stochastic gradient descent
     * 
     * Input:
     *      w   [units]
     *      v_w [units]
     *      d_w [units]
     * Output:
     *      w   [units]
     *      v_w [units]
     * */     
    for (int i = 0; i < units; i++)
    {
        v_w[i] = 0.2 * v_w[i] - 0.8 * LEARNING_RATE * d_w[i];
        CLIP(v_w+i, -1, 1);
        w[i] = w[i] + v_w[i];
    }
}


static void gradient_descent_a(void *argv)
{
    alexnet *net = (alexnet *)argv;
    momentum_sgd(net->fc1.weights, v_fc1_weights, net->fc1.d_weights, C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L);
}

static void gradient_descent_b(void *argv)
{
    alexnet *net = (alexnet *)argv;
    momentum_sgd(net->fc2.weights, v_fc2_weights, net->fc2.d_weights, FC6_LAYER*FC7_LAYER);
}

static void gradient_descent_c(void *argv)
{
    alexnet *net = (alexnet *)argv;
    momentum_sgd(net->fc3.weights, v_fc3_weights, net->fc3.d_weights, FC7_LAYER*OUT_LAYER);
}

static void gradient_descent_d(void *argv)
{
    alexnet *net = (alexnet *)argv;

    momentum_sgd(net->conv1.weights, v_conv1_weights, net->conv1.d_weights, C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L);
    momentum_sgd(net->conv2.weights, v_conv2_weights, net->conv2.d_weights, C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L);
    momentum_sgd(net->conv3.weights, v_conv3_weights, net->conv3.d_weights, C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L);
    momentum_sgd(net->conv4.weights, v_conv4_weights, net->conv4.d_weights, C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L);
    momentum_sgd(net->conv5.weights, v_conv5_weights, net->conv5.d_weights, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L);

    momentum_sgd(net->conv1.bias,   v_conv1_bias,   net->conv1.d_bias, C1_CHANNELS);
    momentum_sgd(net->conv2.bias,   v_conv2_bias,   net->conv2.d_bias, C2_CHANNELS);
    momentum_sgd(net->conv3.bias,   v_conv3_bias,   net->conv3.d_bias, C3_CHANNELS);
    momentum_sgd(net->conv4.bias,   v_conv4_bias,   net->conv4.d_bias, C4_CHANNELS);
    momentum_sgd(net->conv5.bias,   v_conv5_bias,   net->conv5.d_bias, C5_CHANNELS);
    momentum_sgd(net->fc1.bias,     v_fc1_bias,     net->fc1.d_bias,   FC6_LAYER);
    momentum_sgd(net->fc2.bias,     v_fc2_bias,     net->fc2.d_bias,   FC7_LAYER);
    momentum_sgd(net->fc3.bias,     v_fc3_bias,     net->fc3.d_bias,   OUT_LAYER);

    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
}

static void gradient_descent(alexnet *net)
{
    pthread_t tid[4];
    pthread_create(&tid[0], NULL, gradient_descent_a, (void *)(net));
    pthread_create(&tid[1], NULL, gradient_descent_b, (void *)(net));
    pthread_create(&tid[2], NULL, gradient_descent_c, (void *)(net));
    pthread_create(&tid[3], NULL, gradient_descent_d, (void *)(net));
    pthread_join(tid[0], NULL);
    pthread_join(tid[1], NULL);
    pthread_join(tid[2], NULL);
    pthread_join(tid[3], NULL);
}


void calloc_alexnet_d_params(alexnet *net)
{
    calloc_conv_dweights(&(net->conv1));
    calloc_conv_dweights(&(net->conv2));
    calloc_conv_dweights(&(net->conv3));
    calloc_conv_dweights(&(net->conv4));
    calloc_conv_dweights(&(net->conv5));
    calloc_fc_dweights(&(net->fc1));
    calloc_fc_dweights(&(net->fc2));
    calloc_fc_dweights(&(net->fc3));
    calloc_batchnorm_dweights(&(net->bn1));
    calloc_batchnorm_dweights(&(net->bn2));
    calloc_batchnorm_dweights(&(net->bn3));
    calloc_batchnorm_dweights(&(net->bn4));
    calloc_batchnorm_dweights(&(net->bn5));
}

void free_alexnet_d_params(alexnet *net)
{
    free_conv_dweights(&(net->conv1));
    free_conv_dweights(&(net->conv2));
    free_conv_dweights(&(net->conv3));
    free_conv_dweights(&(net->conv4));
    free_conv_dweights(&(net->conv5));
    free_fc_dweights(&(net->fc1));
    free_fc_dweights(&(net->fc2));
    free_fc_dweights(&(net->fc3));

    free_batchnorm_dweights(&(net->bn1));
    free_batchnorm_dweights(&(net->bn2));
    free_batchnorm_dweights(&(net->bn3));
    free_batchnorm_dweights(&(net->bn4));
    free_batchnorm_dweights(&(net->bn5));
}

void backward_alexnet(alexnet *net, int *batch_Y)
{
    /**
     * alexnet backward
     * 
     * Input:
     *      net:      our network
     *      batch_Y:  labels of images
     * Output:
     *      net
     * */
    calloc_alexnet_d_params(net);

    net->fc3.d_output = (float *)calloc(net->fc3.out_units, sizeof(float));
    cross_entropy_loss(net->fc3.d_output, net->output, batch_Y, OUT_LAYER, net->fc3.batchsize);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc3.d_input = (float *)calloc(net->fc3.in_units, sizeof(float));
    fc_op_backward(&(net->fc3));
    free(net->fc3.d_output);
    free(net->fc3.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->fc3)) duration: %.4fs \n", duration);
#endif

    net->relu7.d_input = (float *)calloc(net->relu7.units, sizeof(float));
    net->relu7.d_output = net->fc3.d_input;
    relu_op_backward(&(net->relu7));
    free(net->relu7.d_output);
    free(net->relu7.output);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc2.d_input = (float *)calloc(net->fc2.in_units, sizeof(float));
    net->fc2.d_output = net->relu7.d_input;
    fc_op_backward(&(net->fc2));
    free(net->fc2.d_output);
    free(net->fc2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->fc2)) duration: %.4fs \n", duration);
#endif

    net->relu6.d_input = (float *)calloc(net->relu6.units, sizeof(float));
    net->relu6.d_output = net->fc2.d_input;
    relu_op_backward(&(net->relu6));
    free(net->relu6.d_output);
    free(net->relu6.output);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc1.d_input = (float *)calloc(net->fc1.in_units, sizeof(float));
    net->fc1.d_output = net->relu6.d_input;
    fc_op_backward(&(net->fc1));
    free(net->fc1.d_output);
    free(net->fc1.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->fc1)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp5.d_input = (float *)calloc(net->mp5.in_units, sizeof(float));
    net->mp5.d_output = net->fc1.d_input;
    max_pooling_op_backward(&(net->mp5));
    free(net->mp5.d_output);
    free(net->mp5.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->mp5)) duration: %.4fs \n", duration);
#endif

    net->relu5.d_input = (float *)calloc(net->relu5.units, sizeof(float));
    net->relu5.d_output = net->mp5.d_input;
    relu_op_backward(&(net->relu5));
    free(net->relu5.d_output);
    free(net->relu5.output);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn5.d_input = (float *)calloc(net->bn5.units, sizeof(float));
    net->bn5.d_output = net->relu5.d_input;
    batch_norm_op_backward(&(net->bn5));
    free(net->bn5.d_output);
    free(net->bn5.output);
    for(int i=0; i< net->bn5.units; i++)
    {
        if(net->bn5.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn5  %f\n",net->bn5.d_input[i]);
            exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn5)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv5.d_input = (float *)calloc(net->conv5.in_units, sizeof(float));
    net->conv5.d_output = net->bn5.d_input;
    conv_op_backward(&(net->conv5));
    free(net->conv5.d_output);
    free(net->conv5.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv5)) duration: %.4fs \n", duration);
#endif

    net->relu4.d_input = (float *)calloc(net->relu4.units, sizeof(float));
    net->relu4.d_output = net->conv5.d_input;
    relu_op_backward(&(net->relu4));
    free(net->relu4.d_output);
    free(net->relu4.output);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn4.d_input = (float *)calloc(net->bn4.units, sizeof(float));
    net->bn4.d_output = net->relu4.d_input;
    batch_norm_op_backward(&(net->bn4));
    free(net->bn4.d_output);
    free(net->bn4.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn4)) duration: %.4fs \n", duration);
#endif

    for(int i=0; i< net->bn4.units; i++)
    {
        if(net->bn4.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn4  %f\n",net->bn4.d_input[i]);
            exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv4.d_input = (float *)calloc(net->conv4.in_units, sizeof(float));
    net->conv4.d_output = net->bn4.d_input;
    conv_op_backward(&(net->conv4));
    free(net->conv4.d_output);
    free(net->conv4.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv4)) duration: %.4fs \n", duration);
#endif

    net->relu3.d_input = (float *)calloc(net->relu3.units, sizeof(float));
    net->relu3.d_output = net->conv4.d_input;
    relu_op_backward(&(net->relu3));
    free(net->relu3.d_output);
    free(net->relu3.output);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn3.d_input = (float *)calloc(net->bn3.units, sizeof(float));
    net->bn3.d_output = net->relu3.d_input;
    batch_norm_op_backward(&(net->bn3));
    free(net->bn3.d_output);
    free(net->bn3.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn3)) duration: %.4fs \n", duration);
#endif

    for(int i=0; i< net->bn3.units; i++)
    {
        if(net->bn3.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn3  %f\n",net->bn3.d_input[i]);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv3.d_input = (float *)calloc(net->conv3.in_units, sizeof(float));
    net->conv3.d_output = net->bn3.d_input;
    conv_op_backward(&(net->conv3));
    free(net->conv3.d_output);
    free(net->conv3.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv3)) duration: %.4fs \n", duration);
#endif

    for(int i=0; i< net->conv3.in_units; i++)
    {
        if(net->conv3.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! conv3  %f\n",net->conv3.d_input[i]);
            //exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp2.d_input = (float *)calloc(net->mp2.in_units, sizeof(float));
    net->mp2.d_output = net->conv3.d_input;
    max_pooling_op_backward(&(net->mp2));
    free(net->mp2.d_output);
    free(net->mp2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->mp2)) duration: %.4fs \n", duration);
#endif

    net->relu2.d_input = (float *)calloc(net->relu2.units, sizeof(float));
    net->relu2.d_output = net->mp2.d_input;
    relu_op_backward(&(net->relu2));
    free(net->relu2.d_output);
    free(net->relu2.output);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn2.d_input = (float *)calloc(net->bn2.units, sizeof(float));
    net->bn2.d_output = net->relu2.d_input;
    batch_norm_op_backward(&(net->bn2));
    free(net->bn2.d_output);
    free(net->bn2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn2)) duration: %.4fs \n", duration);
#endif

    for(int i=0; i< net->bn2.units; i++)
    {
        if(net->bn2.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn2  %f\n",net->bn2.d_input[i]);
            //exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv2.d_input = (float *)calloc(net->conv2.in_units, sizeof(float));
    net->conv2.d_output = net->bn2.d_input;
    conv_op_backward(&(net->conv2));
    free(net->conv2.d_output);
    free(net->conv2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv2)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp1.d_input = (float *)calloc(net->mp1.in_units, sizeof(float));
    net->mp1.d_output = net->conv2.d_input;
    max_pooling_op_backward(&(net->mp1));
    free(net->mp1.d_output);
    free(net->mp1.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->mp1)) duration: %.4fs \n", duration);
#endif

    net->relu1.d_input = (float *)calloc(net->relu1.units, sizeof(float));
    net->relu1.d_output = net->mp1.d_input;
    relu_op_backward(&(net->relu1));
    free(net->relu1.d_output);
    free(net->relu1.output);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn1.d_input = (float *)calloc(net->bn1.units, sizeof(float));
    net->bn1.d_output = net->relu1.d_input;
    batch_norm_op_backward(&(net->bn1));
    free(net->bn1.d_output);
    free(net->bn1.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn1)) duration: %.4fs \n", duration);
#endif

    for(int i=0; i< net->bn1.units; i++)
    {
        if(net->bn1.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn1  %f\n",net->bn1.d_input[i]);
            //exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv1.d_input = (float *)calloc(net->conv1.in_units, sizeof(float));
    net->conv1.d_output = net->bn1.d_input;
    conv_op_backward(&(net->conv1));
    free(net->conv1.d_output);
    free(net->conv1.output);
    free(net->conv1.d_input);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv1)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    gradient_descent(net);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward update_params(net) duration: %.4fs \n", duration);
#endif

    free_alexnet_d_params(net);
}


void alexnet_train(alexnet *net, int epochs)
{

    net->input   = (float *)malloc(net->batchsize * net->conv1.in_channels * net->conv1.in_w * net->conv1.in_h * sizeof(float));
    int *batch_Y = (int   *)malloc(net->batchsize * sizeof(int));
    int preds[net->fc3.out_units];
    FILE *fp = fopen("./images.list", "r");

    printf("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>> training begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    struct timespec start, finish; float duration;
    for (int e = 0;e < epochs; e++)
    {
        printf("-----------------------------%d---------------------------------\n", e+1);
        //
        // >>>>>load data<<<<<
        // batch_X <-- images
        // batch_Y <-- labels 
        //
        get_next_batch(net->batchsize, net->input, batch_Y, 
                        net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels, net->fc3.out_units, fp);
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        forward_alexnet(net);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        duration = (finish.tv_sec - start.tv_sec);
        duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        printf("forward_alexnet duration: %.4fs \n", duration);

        for (int i = 0; i < net->batchsize; i++)
            preds[i] = argmax(net->output + i * net->fc3.out_units, net->fc3.out_units);

#ifdef SHOW_PREDCITION_DETAIL
        printf("pred[ ");
        for (int i = 0; i < net->batchsize; i++)
            printf("%d ", preds[i]);
        printf("]  label[ ");
        for (int i = 0; i < net->batchsize; i++)
            printf("%d ", batch_Y[i]);
        printf("]\n");
#endif
        // float metric;
        // metrics(&metric, preds, batch_Y, OUT_LAYER, net->batchsize, METRIC_ACCURACY);
        clock_gettime(CLOCK_MONOTONIC, &start);
        backward_alexnet(net, batch_Y);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        duration = (finish.tv_sec - start.tv_sec);
        duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        printf("backward_alexnet duration: %.4fs \n", duration);
        printf("-----------------------------%d---------------------------------\n", e+1);
    }
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>> training end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    fclose(fp);
    free(net->input);
    free(batch_Y);
}
