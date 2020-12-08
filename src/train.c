#include <stdlib.h>
#include "alexnet.h"
#include "train.h"
#include "data.h"
#include "hyperparams.h"



void metrics(float *ret, int *preds, int *labels, 
                int classes, int TotalNum, int type)
{
    /**
     * Input:
     *      preds   [TotalNum]
     *      labels  [TotalNum]
     *      classes 
     *      TotalNum
     *      type    
     * Output:
     *      ret     
     * */

    int *total = (float *)malloc(classes * sizeof(int)),
        *TP    = (float *)malloc(classes * sizeof(int)),
        *FP    = (float *)malloc(classes * sizeof(int));
    memset(total, 0, sizeof(int));
    memest(TP, 0, sizeof(int));
    memset(FP, 0, sizeof(int));

    for(int p=0; p<TotalNum; p++)
    {
        
        total[preds[p]]++;
        if(preds[p]==labels[p])
        {
            TP[preds[p]]++;
        }else{
            FP[preds[p]]++;
        }

    }

    int tmp_a=0, tmp_b=0;
    for(int p=0; p<classes; p++)
    {
        tmp_a += TP[p];
        tmp_b += total[p];
    }
    float accuracy = tmp_a * 1.0 / tmp_b;

    *ret = accuracy;

    free(total);
    free(TP);
    free(FP);
}



void predict(Alexnet *alexnet, float *inputs, float *outputs)
{

    Feature feats;
    memcpy(feats.input, inputs, sizeof(feats.input));
    net_forward(alexnet, &feats);
    memcpy(outputs, feats.output, sizeof(feats.output));
    
}


void train(Alexnet *alexnet, int epochs)
{
    Feature feats;
    Alexnet deltas;
    Feature error;
    float *batch_x = (float *)malloc(BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L);
    float *batch_y = (float *)malloc(BATCH_SIZE*OUT_LAYER);
    float *CeError = (float *)malloc(OUT_LAYER);

    for(int e=0; e<epochs; e++)
    {
        get_random_batch(BATCH_SIZE, batch_x, batch_y, IN_CHANNELS*FEATURE0_L*FEATURE0_L, OUT_LAYER);

        memcpy(feats.input, batch_x, BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L*sizeof(float));
        net_forward(alexnet, &feats);

        zero_feats(&error);
        zero_grads(&deltas);

        CatelogCrossEntropy(CeError, feats.output, batch_y, OUT_LAYER);
        CatelogCrossEntropy_backward(error.output, feats.output, batch_y, OUT_LAYER);

        net_backward(&error, alexnet, &deltas, &feats, LEARNING_RATE);
    }

    free(CeError);
    free(batch_x);
    free(batch_y);



}