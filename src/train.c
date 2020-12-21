//
// File:        train.c
// Description: Provide train & evaluate functions
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "alexnet.h"
#include "data.h"
#include "hyperparams.h"


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
    int idx = -1;
    float max = -11111111;
    for(int p=0; p<n; p++)
    {
        if(arr[p] > max)
        {
            idx = p;
            max = arr[p];
        }
    }
    assert(idx!=-1);
    return idx;
}


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

    for(int p=0; p<totNum; p++)
    {
        totPred[preds[p]]++;
        totLabel[labels[p]]++;
        if(preds[p]==labels[p])
        {
            TP[preds[p]]++;
        }
    }

    int tmp_a=0, tmp_b=0;
    for(int p=0; p<classes; p++)
    {
        tmp_a += TP[p];
    }
    float accuracy = tmp_a * 1.0 / totNum;

    if(type==METRIC_ACCURACY)
    {
        *ret = accuracy;
        free(totPred);
        free(totLabel);
        free(TP);
        return;
    }

    float precisions[classes];
    float macro_p = 0;
    for(int p=0; p<classes; p++)
    {
        precisions[p] = TP[p] / totLabel[p];
        macro_p += precisions[p];
    }
    macro_p /= classes;

    if(type==METRIC_PRECISION)
    {
        *ret = macro_p;
        free(totPred);
        free(totLabel);
        free(TP);
        return;
    }

    float recalls[classes];
    float macro_r = 0;
    for(int p=0; p<classes; p++)
    {
        recalls[p] = TP[p] / totPred[p];
        macro_r += recalls[p];
    }
    macro_r /= classes;

    if(type==METRIC_RECALL)
    {
        *ret = macro_r;
        free(totPred);
        free(totLabel);
        free(TP);
        return;
    }

    if(type==METRIC_F1SCORE)
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


void alexnet_inference(alexnet *net)
{

}


void alexnet_train(alexnet *net, int epochs)
{

    float   *batch_X = (float *)malloc(BATCH_SIZE * sizeof(float) * IN_CHANNELS*FEATURE0_L*FEATURE0_L);
    int     *batch_Y = (int   *)malloc(BATCH_SIZE * sizeof(int));

    int preds[OUT_LAYER];
    float metric;
    printf("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    struct timespec start, finish; float duration;
    for(int e=0;e <epochs; e++)
    {
        printf("-----------------------------%d---------------------------------\n", e+1);
        //
        // load data
        // batch_X <-- images
        // batch_Y <-- labels 
        //
        get_random_batch(BATCH_SIZE, batch_X, batch_Y, FEATURE0_L, FEATURE0_L, IN_CHANNELS, OUT_LAYER);
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        net->input = batch_X;
        alexnet_forward(net);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        duration = (finish.tv_sec - start.tv_sec);
        duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        printf("alexnet_forward duration: %.2fs \n", duration);

        for(int i=0; i<BATCH_SIZE; i++)
            preds[i] = argmax(net->output+i*OUT_LAYER, OUT_LAYER);

#ifdef SHOW_PREDCITION_DETAIL
        printf("pred[ ");
        for(int i=0; i<BATCH_SIZE; i++)
        {
            printf("%d ", preds[i]);
        }
        printf("]  label[ ");
        for(int i=0; i<BATCH_SIZE; i++)
        {
            printf("%d ", batch_Y[i]);
        }
        printf("]\n");
#endif
        //metrics(&metric, preds, batch_Y, OUT_LAYER, BATCH_SIZE, METRIC_ACCURACY);
        clock_gettime(CLOCK_MONOTONIC, &start);
        alexnet_backward(net, batch_Y);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        duration = (finish.tv_sec - start.tv_sec);
        duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        printf("alexnet_backward duration: %.2fs \n", duration);
        printf("-----------------------------%d---------------------------------\n\n", e+1);
    }
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    free(batch_X);
    free(batch_Y);
}
