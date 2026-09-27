//
// File:        dropout_layer.c
// Description: Implementation of dropout layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "dropout_layer.h"

void dropout_forward(float *x, float *mask, float prob, int n)
{
    /**
     * PyTorch-style (inverted) dropout regularization, forward pass
     *
     * Input:
     *      x       [n]
     *      prob    prob~(0,1)
     * Output:
     *      x       [n]     scaled in place
     *      mask    [n]     1/keep for kept units, 0 for dropped -- reused by
     *                      dropout_backward on the matching batch
     * */
    if (prob <= 0.0f) {
        for (int i = 0; i < n; i++) mask[i] = 1.0f;
        return;
    }

    float keep_prob = 1.0f - prob;
    float scale = 1.0f / keep_prob;
    for (int i = 0; i < n; i++)
    {
        if (rand()%100 < prob*100) {
            mask[i] = 0.0f;
            x[i] = 0.0f;
        } else {
            mask[i] = scale;
            x[i] *= scale;
        }
    }
}

void dropout_backward(float *dx, const float *mask, int n)
{
    /**
     * dropout backward: same mask the forward pass produced, applied to the
     * incoming gradient (dropped units get zero gradient, kept units get
     * the same 1/keep rescale).
     * */
    for (int i = 0; i < n; i++)
        dx[i] *= mask[i];
}
