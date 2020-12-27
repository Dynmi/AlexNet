//
// File:        dropout_layer.c
// Description: Implementation of dropout layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "dropout_layer.h"

void dropout(float *x, float prob, int units)
{
    /**
     * dropout regularization
     * 
     * Input:
     *      x   [units]
     *      prob    prob~(0,1)
     * Output:
     *      x   [units]
     * */
    for (int i = 0; i < units; i++)
    {
        if (rand()%100 < prob*100)
            x[i] = 0;
    }
}
