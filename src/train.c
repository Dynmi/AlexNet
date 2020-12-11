//
// File:        train.c
// Description: Provide train & evaluate functions
// Author:      Haris Wang
//

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "alexnet.h"
#include "data.h"
#include "hyperparams.h"


int argmax(float *arr, int shift, int n)
{
    /**
     * Return the index of max-value among arr[shift]~arr[shift+n]
     * 
     * Input:
     *      arr
     * Output:
     * Return:
     *      the index of max-value
     * */ 
    int idx=-1,
        max=0;
    for(int p=0; p<n; p++)
    {
        if(arr[shift+p]>max)
        {
            idx = p;
            max = arr[shift+p];
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

    int *totPred  = (float *)malloc(classes * sizeof(int)),
        *totLabel = (float *)malloc(classes * sizeof(int)),
        *TP       = (float *)malloc(classes * sizeof(int));
    memset(totPred, 0, sizeof(int));
    memset(totLabel, 0, sizeof(int));
    memset(TP, 0, sizeof(int));

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
        return;
    }

    if(type==METRIC_F1SCORE)
    {
        *ret = 2*macro_p*macro_r / (macro_p+macro_r);
        return;
    }

    free(totPred);
    free(totLabel);
    free(TP);
}


void predict(Alexnet *alexnet, float *inputs, float *outputs)
{
    /**
     * Make prediction with weights and inputs
     * 
     * Input:
     *      alexnet
     *      inputs      [BATCH_SIZE, IN_CHANNELS*FEATURE0_L*FEATURE0_L]
     * Output:
     *      outputs     [BATCH_SIZE, OUT_LAYER]
     * */

    Feature feats;
    memcpy(feats.input, inputs, sizeof(feats.input));
    net_forward(alexnet, &feats);
    memcpy(outputs, feats.output, sizeof(feats.output));
    
}


void train(Alexnet *alexnet, int epochs)
{
    /**
     * train your model
     * 
     * Input:
     *      alexnet
     *      epochs
     * Output:
     *      alexnet
     * */
    static Feature feats;
    static Alexnet deltas;
    static Feature error;
    float *batch_x = (float *)malloc(BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L*sizeof(float));
    float *batch_y = (float *)malloc(BATCH_SIZE*OUT_LAYER*sizeof(float));
    float *CeError = (float *)malloc(OUT_LAYER*sizeof(float));

    float metric=0; 
    static int labels[BATCH_SIZE],preds[BATCH_SIZE];

    for(int e=0; e<epochs; e++)
    {
        zero_feats(&error);
        zero_grads(&deltas);

        // sample random batch of data for training
        get_random_batch(BATCH_SIZE, batch_x, batch_y, FEATURE0_L, FEATURE0_L, IN_CHANNELS, OUT_LAYER);

        memcpy(feats.input, batch_x, BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L*sizeof(float));
        //printf("after \"memcpy\" \n");

        net_forward(alexnet, &feats);
        //printf("after \"net_forward\" \n");

        // compute CatelogCrossEntropy
        CatelogCrossEntropy(CeError, feats.output, batch_y, OUT_LAYER);
        CatelogCrossEntropy_backward(error.output, feats.output, batch_y, OUT_LAYER);
        //printf("after \"CatelogCrossEntropy_backward\" \n");

        // update all trainable parameters
        net_backward(&error, alexnet, &deltas, &feats, LEARNING_RATE);
        //printf("after \"net_backward\" \n");

        for(int i=0; i<BATCH_SIZE; i++)
        {
            labels[i] = argmax(&(feats.output), i*OUT_LAYER, OUT_LAYER); 
            preds[i]  = argmax(&(feats.output), i*OUT_LAYER, OUT_LAYER);
        }
        metrics(&metric, preds, labels, OUT_LAYER, BATCH_SIZE, METRIC_ACCURACY);
        //printf("At epoch %d, Accuracy on training data is %.2f \n", e, metric);
    
    }

    free(CeError);
    free(batch_x);
    free(batch_y);

}
