#include <stdlib.h>
#include "alexnet.h"
#include "train.h"
#include "data.h"
#include "hyperparams.h"


void predict(Alexnet *alexnet, float *inputs, float *outputs)
{

    Feature feats;
    memcpy(feats.input, inputs, sizeof(feats.input));
    net_forward(alexnet, &feats);
    memcpy(outputs, feats.output, sizeof(feats.output));
    
}


void train(Alexnet *alexnet)
{
    Feature feats;
    float *batch_x = (float *)malloc(4*BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L);
    float *batch_y = (float *)malloc(4*BATCH_SIZE*OUT_LAYER);

    /** 
     * sample batch data
     * */

    net_forward(alexnet, &feats);

    Alexnet deltas;
    Feature error;
    zero_feats(&error);

    compute_mse_error(error.output, feats.output, batch_y, OUT_LAYER);

    zero_grads(&deltas);
    net_backward(&error, alexnet, &deltas, &feats, LEARNING_RATE);

    free(batch_x);
    free(batch_y);



}