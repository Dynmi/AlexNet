//
// File:        inference.c
// Description: Implementation of inference function
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "alexnet.h"
#include "data.h"

void alexnet_inference(alexnet *net, char *filename)
{
    image img;
    make_image(&img, FEATURE0_L, FEATURE0_L, IN_CHANNELS);
    img = load_image(filename, FEATURE0_L, FEATURE0_L, IN_CHANNELS, 0);
    net->input = img.data;    
    forward_alexnet(net);
    int pred = argmax(net->output, OUT_LAYER);
    printf("prediction: %d", pred);
    free_image(&img);
}
