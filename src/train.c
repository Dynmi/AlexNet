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


void train(Alexnet *alexnet, int epochs)
{
    Feature feats;
    Alexnet deltas;
    Feature error;
    float *batch_x = (float *)malloc(BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L);
    float *batch_y = (float *)malloc(BATCH_SIZE*OUT_LAYER);

    for(int e=0; e<epochs; e++)
    {
        get_random_batch(BATCH_SIZE, batch_x, batch_y, IN_CHANNELS*FEATURE0_L*FEATURE0_L, OUT_LAYER);

        memcpy(feats.input, batch_x, BATCH_SIZE*IN_CHANNELS*FEATURE0_L*FEATURE0_L*sizeof(float));
        net_forward(alexnet, &feats);

        zero_feats(&error);
        zero_grads(&deltas);
        compute_mse_error(error.output, feats.output, batch_y, OUT_LAYER);
        net_backward(&error, alexnet, &deltas, &feats, LEARNING_RATE);
    }

    free(batch_x);
    free(batch_y);



}