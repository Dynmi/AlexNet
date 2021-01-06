//
// File:        alexnet.c
// Description: alexnet.c
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "alexnet.h"


void metrics(float *ret, int *preds, int *labels, 
                int classes, int totNum, int type)
{
    /**
     * Compute metric on 'preds' and 'labels'
     * 
     * Input:
     *      preds   [totNum]
     *      labels  [totNum]
     *      classes 
     *      totNum
     *      type    
     * Output:
     *      ret     
     * */

    int *totPred  = (int *)malloc(classes * sizeof(int)),
        *totLabel = (int *)malloc(classes * sizeof(int)),
        *TP       = (int *)malloc(classes * sizeof(int));
    memset(totPred, 0, classes * sizeof(int));
    memset(totLabel, 0, classes * sizeof(int));
    memset(TP, 0, classes * sizeof(int));

    for (int p = 0; p < totNum; p++)
    {
        totPred[preds[p]]++;
        totLabel[labels[p]]++;
        if(preds[p] == labels[p])
        {
            TP[preds[p]]++;
        }
    }

    int tmp_a=0, tmp_b=0;
    for (int p =0 ; p < classes; p++)
    {
        tmp_a += TP[p];
    }
    float accuracy = tmp_a * 1.0 / totNum;

    if (type == METRIC_ACCURACY)
    {
        *ret = accuracy;
        free(totPred);
        free(totLabel);
        free(TP);
        return;
    }

    float precisions[classes];
    float macro_p = 0;
    for (int p = 0; p < classes; p++)
    {
        precisions[p] = TP[p] / totLabel[p];
        macro_p += precisions[p];
    }
    macro_p /= classes;

    if (type == METRIC_PRECISION)
    {
        *ret = macro_p;
        free(totPred);
        free(totLabel);
        free(TP);
        return;
    }

    float recalls[classes];
    float macro_r = 0;
    for (int p = 0; p < classes; p++)
    {
        recalls[p] = TP[p] / totPred[p];
        macro_r += recalls[p];
    }
    macro_r /= classes;

    if (type == METRIC_RECALL)
    {
        *ret = macro_r;
        free(totPred);
        free(totLabel);
        free(TP);
        return;
    }

    if (type == METRIC_F1SCORE)
    {
        *ret = 2*macro_p*macro_r / (macro_p+macro_r);
        free(totPred);
        free(totLabel);
        free(TP);
        return;
    }

    free(totPred);
    free(totLabel);
    free(TP);
}

int argmax(float *arr, int n)
{
    /**
     * Return the index of max-value among arr ~ arr+n
     * 
     * Input:
     *      arr
     * Output:
     * Return:
     *      the index of max-value
     * */ 
    int   idx = -1;
    float max = -1111111111;
    for (int p = 0; p<n; p++)
    {
        if (arr[p] > max)
        {
            idx = p;
            max = arr[p];
        }
    }
    assert(idx!=-1);
    return idx;
}


static struct timespec start, finish; 
static float duration;
void forward_alexnet(alexnet *net)
{
    net->conv1.input = net->input;

    //printf(">>>>>>>>>>>>>>>>>>conv1>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv1.output = (float *)calloc(net->batchsize * net->conv1.out_units, sizeof(float));
    conv_op_forward(&(net->conv1));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv1)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn1.output = (float *)malloc(net->batchsize * sizeof(float) * net->bn1.units);
    net->bn1.input = net->conv1.output;
    batch_norm_op_forward(&(net->bn1));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn1)) duration: %.4fs \n", duration);
#endif

    net->relu1.output = (float *)malloc(net->batchsize * sizeof(float) * net->relu1.units);
    net->relu1.input = net->bn1.output;
    relu_op_forward(&(net->relu1));

    net->mp1.output = (float *)malloc(net->batchsize * sizeof(float) * net->mp1.out_units);
    net->mp1.input = net->relu1.output;
    max_pooling_op_forward(&(net->mp1));

    //printf(">>>>>>>>>>>>>>>>>>conv2>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv2.output = (float *)calloc(net->batchsize * net->conv2.out_units, sizeof(float));
    net->conv2.input = net->mp1.output;
    conv_op_forward(&(net->conv2));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv2)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn2.output = (float *)malloc(net->batchsize * sizeof(float) * net->bn2.units);
    net->bn2.input = net->conv2.output;
    batch_norm_op_forward(&(net->bn2));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn2)) duration: %.4fs \n", duration);
#endif

    net->relu2.output = (float *)malloc(net->batchsize * sizeof(float) * net->relu2.units);
    net->relu2.input = net->bn2.output;
    relu_op_forward(&(net->relu2));

    net->mp2.output = (float *)malloc(net->batchsize * sizeof(float) * net->mp2.out_units);
    net->mp2.input = net->relu2.output;
    max_pooling_op_forward(&(net->mp2));

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    //printf(">>>>>>>>>>>>>>>>>>conv3>>>>>>>>>>>>>>>>>>>>>>>>> \n");
    net->conv3.output = (float *)calloc(net->batchsize * net->conv3.out_units, sizeof(float));
    net->conv3.input = net->mp2.output;
    conv_op_forward(&(net->conv3));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv3)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn3.output = (float *)malloc(net->batchsize * sizeof(float) * net->bn3.units);
    net->bn3.input = net->conv3.output;
    batch_norm_op_forward(&(net->bn3));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn3)); duration: %.4fs \n", duration);
#endif

    net->relu3.output = (float *)malloc(net->batchsize * sizeof(float) * net->relu3.units);
    net->relu3.input = net->bn3.output;
    relu_op_forward(&(net->relu3));

    //printf(">>>>>>>>>>>>>>>>>>conv4>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv4.output = (float *)calloc(net->batchsize * net->conv4.out_units, sizeof(float));
    net->conv4.input = net->relu3.output;
    conv_op_forward(&(net->conv4));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv4)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn4.output = (float *)malloc(net->batchsize * sizeof(float) * net->bn4.units);
    net->bn4.input = net->conv4.output;
    batch_norm_op_forward(&(net->bn4));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn4)) duration: %.4fs \n", duration);
#endif

    net->relu4.output = (float *)malloc(net->batchsize * sizeof(float) * net->relu4.units);
    net->relu4.input = net->bn4.output;
    relu_op_forward(&(net->relu4));

    //printf(">>>>>>>>>>>>>>>>>>conv5>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv5.output = (float *)calloc(net->batchsize * net->conv5.out_units, sizeof(float));
    net->conv5.input = net->relu4.output;
    conv_op_forward(&(net->conv5));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv5)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn5.output = (float *)malloc(net->batchsize * sizeof(float) * net->bn5.units);
    net->bn5.input = net->conv5.output;
    batch_norm_op_forward(&(net->bn5));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn5)) duration: %.4fs \n", duration);
#endif

    net->relu5.output = (float *)malloc(net->batchsize * sizeof(float) * net->relu5.units);
    net->relu5.input = net->bn5.output;
    relu_op_forward(&(net->relu5));

    net->mp5.output = (float *)malloc(net->batchsize * sizeof(float) * net->mp5.out_units);
    net->mp5.input = net->relu5.output;
    max_pooling_op_forward(&(net->mp5));


    dropout(net->mp5.output, DROPOUT_PROB, net->mp5.batchsize * net->mp5.out_units);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc1.output = (float *)calloc(net->batchsize * net->fc1.out_units, sizeof(float));
    net->fc1.input = net->mp5.output;
    fc_op_forward(&(net->fc1));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc1)) duration: %.4fs \n", duration);
#endif

    net->relu6.output = (float *)malloc(net->batchsize * sizeof(float) * net->relu6.units);
    net->relu6.input = net->fc1.output;
    relu_op_forward(&(net->relu6));

    dropout(net->relu6.output, DROPOUT_PROB, net->relu6.batchsize * net->relu6.units);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc2.output = (float *)calloc(net->batchsize * net->fc2.out_units, sizeof(float));
    net->fc2.input = net->fc1.output;
    fc_op_forward(&(net->fc2));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc2)) duration: %.4fs \n", duration);
#endif

    for(int p=0; p< net->fc2.out_units * net->batchsize; p++)
    {
        if(net->fc2.output[p]<(0-64) | net->fc2.output[p]>64)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: fc2 too big/small !!! %f \n", net->fc2.output[p]);
            break;
        }
    }

    net->relu7.output = (float *)malloc(net->batchsize * sizeof(float) * net->relu7.units);
    net->relu7.input = net->fc2.output;
    relu_op_forward(&(net->relu7));

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc3.output = (float *)calloc(net->batchsize * net->fc3.out_units, sizeof(float));
    net->fc3.input = net->fc2.output;
    fc_op_forward(&(net->fc3));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc3) duration: %.4fs \n", duration);
#endif

    net->output = net->fc3.output;
}


void malloc_alexnet(alexnet *net)
{
    calloc_conv_weights(&(net->conv1));
    calloc_conv_weights(&(net->conv2));
    calloc_conv_weights(&(net->conv3));
    calloc_conv_weights(&(net->conv4));
    calloc_conv_weights(&(net->conv5));
    calloc_fc_weights(&(net->fc1));
    calloc_fc_weights(&(net->fc2));
    calloc_fc_weights(&(net->fc3));

    calloc_batchnorm_weights(&(net->bn1));
    calloc_batchnorm_weights(&(net->bn2));
    calloc_batchnorm_weights(&(net->bn3));
    calloc_batchnorm_weights(&(net->bn4));
    calloc_batchnorm_weights(&(net->bn5));
}

void free_alexnet(alexnet *net)
{
    free_conv_weights(&(net->conv1));
    free_conv_weights(&(net->conv2));
    free_conv_weights(&(net->conv3));
    free_conv_weights(&(net->conv4));
    free_conv_weights(&(net->conv5));
    free_fc_weights(&(net->fc1));
    free_fc_weights(&(net->fc2));
    free_fc_weights(&(net->fc3));

    free_batchnorm_weights(&(net->bn1));
    free_batchnorm_weights(&(net->bn2));
    free_batchnorm_weights(&(net->bn3));
    free_batchnorm_weights(&(net->bn4));
    free_batchnorm_weights(&(net->bn5));
}

static inline float U_Random(void)
{
    srand( (unsigned)time( NULL ) );
    return (float)(rand() % 100) / 100;
}

static void gauss_initialization(float *p, int n, int in_units, int out_units)
{
    float mean  = 0;
    float stddv = 0.01;

	float V1, V2, S, X;
	static int phase = 0;
    for (int shift = 0; shift < n; shift++)
    {
        if (phase == 0) {
            do {
                float U1 = (float) rand() / RAND_MAX;
                float U2 = (float) rand() / RAND_MAX;

                V1 = 2 * U1 - 1;
                V2 = 2 * U2 - 1;
                S = V1 * V1 + V2 * V2;
            } while (S >= 1 || S == 0);
    
            X = V1 * sqrt(-2 * log(S) / S);
        }else {
            X = V2 * sqrt(-2 * log(S) / S);
        }
        phase = 1 - phase;

        p[shift] = mean + stddv * X;
    }
}


//
// save trainable weights of network
//
void save_alexnet(alexnet *net, char *filename)
{
    /**
     * save weights of net to file
     * */
    FILE *fp = fopen(filename, "wb");
    save_conv_weights(&(net->conv1), fp);
    save_conv_weights(&(net->conv2), fp);
    save_conv_weights(&(net->conv3), fp);
    save_conv_weights(&(net->conv4), fp);
    save_conv_weights(&(net->conv5), fp);
    save_fc_weights(&(net->fc1), fp);
    save_fc_weights(&(net->fc2), fp);
    save_fc_weights(&(net->fc3), fp);
    save_batchnorm_weights(&(net->bn1), fp);
    save_batchnorm_weights(&(net->bn2), fp);
    save_batchnorm_weights(&(net->bn3), fp);
    save_batchnorm_weights(&(net->bn4), fp);
    save_batchnorm_weights(&(net->bn5), fp);
    fclose(fp);
    printf("Save weights to \"%s\" successfully... \n", filename);
}

//
// load trainable weights of network
//
void load_alexnet(alexnet *net, char *filename)
{
    /**
     * load weights of network from file
     * */
    FILE *fp = fopen(filename, "rb");
    load_conv_weights(&(net->conv1), fp);
    load_conv_weights(&(net->conv2), fp);
    load_conv_weights(&(net->conv3), fp);
    load_conv_weights(&(net->conv4), fp);
    load_conv_weights(&(net->conv5), fp);
    load_fc_weights(&(net->fc1), fp);
    load_fc_weights(&(net->fc2), fp);
    load_fc_weights(&(net->fc3), fp);
    load_batchnorm_weights(&(net->bn1), fp);
    load_batchnorm_weights(&(net->bn2), fp);
    load_batchnorm_weights(&(net->bn3), fp);
    load_batchnorm_weights(&(net->bn4), fp);
    load_batchnorm_weights(&(net->bn5), fp);
    fclose(fp);
    printf("Load weights from \"%s\" successfully... \n", filename);
}


void setup_alexnet(alexnet *net, short batchsize)
{
    /**
     * initialize alexnet
     * */
    net->batchsize = batchsize;
    net->conv1.batchsize = batchsize;
    net->conv2.batchsize = batchsize;
    net->conv3.batchsize = batchsize;
    net->conv4.batchsize = batchsize;
    net->conv5.batchsize = batchsize;
    net->fc1.batchsize   = batchsize;
    net->fc2.batchsize   = batchsize;
    net->fc3.batchsize   = batchsize;
    net->bn1.batchsize   = batchsize;
    net->bn2.batchsize   = batchsize;
    net->bn3.batchsize   = batchsize;
    net->bn4.batchsize   = batchsize;
    net->bn5.batchsize   = batchsize;
    net->mp1.batchsize   = batchsize;
    net->mp2.batchsize   = batchsize;
    net->mp5.batchsize   = batchsize;
    net->relu1.batchsize = batchsize;
    net->relu2.batchsize = batchsize;
    net->relu3.batchsize = batchsize;
    net->relu4.batchsize = batchsize;
    net->relu5.batchsize = batchsize;
    net->relu6.batchsize = batchsize;
    net->relu7.batchsize = batchsize;

    net->conv1.in_channels = IN_CHANNELS;
    net->conv1.out_channels = C1_CHANNELS;
    net->conv1.in_h = FEATURE0_L;
    net->conv1.in_w = FEATURE0_L;
    net->conv1.kernel_size = C1_KERNEL_L;
    net->conv1.padding = C1_PADDING;
    net->conv1.stride = C1_STRIDES;
    net->conv1.out_h = FEATURE1_L;
    net->conv1.out_w = FEATURE1_L;
    net->conv1.in_units = IN_CHANNELS*FEATURE0_L*FEATURE0_L;
    net->conv1.out_units = C1_CHANNELS*FEATURE1_L*FEATURE1_L;

    net->bn1.units = net->conv1.out_units;

    net->relu1.units = net->bn1.units;
    
    net->mp1.channels = C1_CHANNELS;
    net->mp1.stride = 2;
    net->mp1.kernel_size = 3;
    net->mp1.in_h = FEATURE1_L;
    net->mp1.in_w = FEATURE1_L;
    net->mp1.out_w = POOLING1_L;
    net->mp1.out_h = POOLING1_L;
    net->mp1.in_units = net->relu1.units;
    net->mp1.out_units = C1_CHANNELS*POOLING1_L*POOLING1_L;

    net->conv2.in_channels = C1_CHANNELS;
    net->conv2.out_channels = C2_CHANNELS;
    net->conv2.in_h = POOLING1_L;
    net->conv2.in_w = POOLING1_L;
    net->conv2.kernel_size = C2_KERNEL_L;
    net->conv2.padding = C2_PADDING;
    net->conv2.stride = C2_STRIDES;
    net->conv2.out_h = FEATURE2_L;
    net->conv2.out_w = FEATURE2_L;
    net->conv2.in_units = net->mp1.out_units;
    net->conv2.out_units = C2_CHANNELS*FEATURE2_L*FEATURE2_L;

    net->bn2.units = net->conv2.out_units;

    net->relu2.units = net->bn2.units;

    net->mp2.channels = C2_CHANNELS;
    net->mp2.stride = 2;
    net->mp2.kernel_size = 3;
    net->mp2.in_h = FEATURE2_L;
    net->mp2.in_w = FEATURE2_L;
    net->mp2.out_w = POOLING2_L;
    net->mp2.out_h = POOLING2_L;
    net->mp2.in_units = net->relu2.units;
    net->mp2.out_units = C2_CHANNELS*POOLING2_L*POOLING2_L;

    net->conv3.in_channels = C2_CHANNELS;
    net->conv3.out_channels = C3_CHANNELS;
    net->conv3.in_h = POOLING2_L;
    net->conv3.in_w = POOLING2_L;
    net->conv3.kernel_size = C3_KERNEL_L;
    net->conv3.padding = C3_PADDING;
    net->conv3.stride = C3_STRIDES;
    net->conv3.out_h = FEATURE3_L;
    net->conv3.out_w = FEATURE3_L;
    net->conv3.in_units = net->mp2.out_units;
    net->conv3.out_units = C3_CHANNELS*FEATURE3_L*FEATURE3_L;

    net->bn3.units = net->conv3.out_units;

    net->relu3.units = net->bn3.units;

    net->conv4.in_channels = C3_CHANNELS;
    net->conv4.out_channels = C4_CHANNELS;
    net->conv4.in_h = FEATURE3_L;
    net->conv4.in_w = FEATURE3_L;
    net->conv4.kernel_size = C4_KERNEL_L;
    net->conv4.padding = C4_PADDING;
    net->conv4.stride = C4_STRIDES;
    net->conv4.out_h = FEATURE4_L;
    net->conv4.out_w = FEATURE4_L;
    net->conv4.in_units = net->relu3.units;
    net->conv4.out_units = C4_CHANNELS*FEATURE4_L*FEATURE4_L;

    net->bn4.units = net->conv4.out_units;

    net->relu4.units = net->bn4.units;

    net->conv5.in_channels = C4_CHANNELS;
    net->conv5.out_channels = C5_CHANNELS;
    net->conv5.in_h = FEATURE5_L;
    net->conv5.in_w = FEATURE5_L;
    net->conv5.kernel_size = C5_KERNEL_L;
    net->conv5.padding = C5_PADDING;
    net->conv5.stride = C5_STRIDES;
    net->conv5.out_h = FEATURE5_L;
    net->conv5.out_w = FEATURE5_L;
    net->conv5.in_units = net->relu4.units;
    net->conv5.out_units = C5_CHANNELS*FEATURE5_L*FEATURE5_L;

    net->bn5.units = net->conv5.out_units;

    net->relu5.units = net->bn5.units;

    net->mp5.channels = C5_CHANNELS;
    net->mp5.stride = 2;
    net->mp5.kernel_size = 3;
    net->mp5.in_h = FEATURE5_L;
    net->mp5.in_w = FEATURE5_L;
    net->mp5.out_w = POOLING5_L;
    net->mp5.out_h = POOLING5_L;
    net->mp5.in_units = net->relu5.units;
    net->mp5.out_units = C5_CHANNELS*POOLING5_L*POOLING5_L;

    net->fc1.in_units = net->mp5.out_units;
    net->fc1.out_units = FC6_LAYER;
    
    net->relu6.units = FC6_LAYER; 
    
    net->fc2.in_units = FC6_LAYER;
    net->fc2.out_units = FC7_LAYER;

    net->relu7.units = FC7_LAYER;

    net->fc3.in_units = FC7_LAYER;
    net->fc3.out_units = OUT_LAYER;
}

void alexnet_init_weights(alexnet *net, char *weights_path)
{

    if(weights_path != NULL) // load a pre-trained model
    {
        load_alexnet(net, weights_path);
        return;
    }

    // initialize weights for this network
    gauss_initialization(net->conv1.weights, C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L, net->conv1.in_units, net->conv1.out_units);
    gauss_initialization(net->conv2.weights, C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L, net->conv2.in_units, net->conv2.out_units);
    gauss_initialization(net->conv3.weights, C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L, net->conv3.in_units, net->conv3.out_units);
    gauss_initialization(net->conv4.weights, C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L, net->conv4.in_units, net->conv4.out_units);
    gauss_initialization(net->conv5.weights, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L, net->conv5.in_units, net->conv5.out_units);
    gauss_initialization(net->fc1.weights, C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L, net->fc1.in_units, net->fc1.out_units);
    gauss_initialization(net->fc2.weights, FC6_LAYER*FC7_LAYER, net->fc2.in_units, net->fc2.out_units);
    gauss_initialization(net->fc3.weights, FC7_LAYER*OUT_LAYER, net->fc3.in_units, net->fc3.out_units);

    int i;
    for(i=0; i<C1_CHANNELS; i++)
        net->conv1.bias[i] = 1;
    for(i=0; i<C2_CHANNELS; i++)
        net->conv2.bias[i] = 1;
    for(i=0; i<C3_CHANNELS; i++)
        net->conv3.bias[i] = 1;
    for(i=0; i<C4_CHANNELS; i++)
        net->conv4.bias[i] = 1;
    for(i=0; i<C5_CHANNELS; i++)
        net->conv5.bias[i] = 1;
    for(i=0; i<FC6_LAYER; i++)
        net->fc1.bias[i] = 1;
    for(i=0; i<FC7_LAYER; i++)
        net->fc2.bias[i] = 1;
    for(i=0; i<OUT_LAYER; i++)
        net->fc3.bias[i] = 1;

    for(i=0; i<(net->bn1.units); i++)
        net->bn1.gamma[i] = 1;
    for(i=0; i<(net->bn2.units); i++)
        net->bn2.gamma[i] = 1;
    for(i=0; i<(net->bn3.units); i++)
        net->bn3.gamma[i] = 1;
    for(i=0; i<(net->bn4.units); i++)
        net->bn4.gamma[i] = 1;
    for(i=0; i<(net->bn5.units); i++)
        net->bn5.gamma[i] = 1;
    
    memset(net->bn1.beta, 0, sizeof(float)*(net->bn1.units));
    memset(net->bn2.beta, 0, sizeof(float)*(net->bn2.units));
    memset(net->bn3.beta, 0, sizeof(float)*(net->bn3.units));
    memset(net->bn4.beta, 0, sizeof(float)*(net->bn4.units));
    memset(net->bn5.beta, 0, sizeof(float)*(net->bn5.units));
}


int main(int argc, char* argv[])
{
    /**
     * 
     * Entrance
     * 
     * */
    static alexnet net;
    
    if (0 == strcmp(argv[1], "train"))
    {
        // 
        // $./alexnet train -batchsize 4 -epochs 1000 -load_pretrained ./in.weights -save ./out.weights
        //
        int     batchsize = 8, 
                epochs = 1000;
        short   is_save=0, 
                is_load=0;
        char    weights_in_path[256];
        char    weights_out_path[256]; 
        for (int i = 2; i < argc-1; i++)
        {
            if (0 == strcmp(argv[i], "-batchsize"))
                sscanf(argv[i+1], "%d", &batchsize);
            if (0 == strcmp(argv[i], "-epochs"))
                sscanf(argv[i+1], "%d", &epochs);
            if (0 == strcmp(argv[i], "-load_pretrained"))
            {
                is_load=1;
                sprintf(weights_in_path, "%s", argv[i+1]);
            }
            if (0 == strcmp(argv[i], "-save"))
            {
                is_save=1;
                sprintf(weights_out_path, "%s", argv[i+1]);
            }
        }

        printf("batch size: %d \n", batchsize);
        printf("epochs: %d \n", epochs);
        setup_alexnet(&net, batchsize);
        malloc_alexnet(&net);    
        if (is_load) {
            alexnet_init_weights(&net, weights_in_path);
        }else {
            alexnet_init_weights(&net, NULL);
        }
        alexnet_train(&net, epochs);
        if (is_save)
            save_alexnet(&net, weights_out_path);
        free_alexnet(&net);

    }else if (0 == strcmp(argv[1], "inference"))
    {
        //
        // $./alexnet inference -input ./data/0001.jpeg -load ./a1.weights
        //
        char img_path[256];
        char weights_path[256];
        for (int i = 2; i < argc-1; i++)
        {
            if (0 == strcmp(argv[i], "-input"))
                sprintf(img_path, "%s", argv[i+1]);
            if (0 == strcmp(argv[i], "-load"))
                sprintf(weights_path, "%s", argv[i+1]);
        }
        setup_alexnet(&net, 1);
        malloc_alexnet(&net);    
        alexnet_init_weights(&net, weights_path);
        printf("alexnet setup fininshed. Waiting for inference...\n");
        alexnet_inference(&net, argv[2]);
        free_alexnet(&net);
    }else {
        printf("Error: Unknown argument!!! \n");
    }
}
