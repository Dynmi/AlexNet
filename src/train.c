#include <stdlib.h>
#include <stdio.h>
#include "alexnet.h"
#include "train.h"
#include "data.h"
#include "hyperparams.h"


int argmax(float *inputs, int units)
{
    int idx=-1,
        max=0;
    for(int p=0; p<units; p++)
    {
        if(inputs[p]>max)
        {
            idx = p;
            max = inputs[p];
        }
    }
    
    assert(idx!=-1);
    return idx;
}


void metrics(float *ret, int *preds, int *labels, 
                int classes, int totNum, int type)
{
    /**
     * Compute metric on given data
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
    memest(TP, 0, sizeof(int));

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
     * Implementation of inference 
     * 
     * Input:
     *      alexnet
     *      inputs
     * Output:
     *      outputs
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
    Feature feats;
    Alexnet deltas;
    Feature error;
    float *batch_x = (float *)malloc(BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L);
    float *batch_y = (float *)malloc(BATCH_SIZE*OUT_LAYER);
    float *CeError = (float *)malloc(OUT_LAYER);

    for(int e=0; e<epochs; e++)
    {
        zero_feats(&error);
        zero_grads(&deltas);

        // sample random batch of data for training
        get_random_batch(BATCH_SIZE, batch_x, batch_y, IN_CHANNELS*FEATURE0_L*FEATURE0_L, OUT_LAYER);
        memcpy(feats.input, batch_x, BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L*sizeof(float));

        net_forward(alexnet, &feats);

        // compute CatelogCrossEntropy
        CatelogCrossEntropy(CeError, feats.output, batch_y, OUT_LAYER);
        CatelogCrossEntropy_backward(error.output, feats.output, batch_y, OUT_LAYER);

        // update all trainable parameters
        net_backward(&error, alexnet, &deltas, &feats, LEARNING_RATE);

        static float metric=0; 
        static int labels[BATCH_SIZE],preds[BATCH_SIZE];
        for(int i=0; i<BATCH_SIZE; i++)
        {
            labels[i] = argmax(feats.output[i*OUT_LAYER], OUT_LAYER); 
            preds[i]  = argmax(feats.output[i*OUT_LAYER], OUT_LAYER);
        }
        metrics(&metric, preds, labels, OUT_LAYER, BATCH_SIZE, METRIC_ACCURACY);
        printf("At epoch %d, Accuracy on training data is %d \n", e, metric);
    }

    free(CeError);
    free(batch_x);
    free(batch_y);

}