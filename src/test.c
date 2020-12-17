#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "alexnet.h"

int main(void)
{
    static alexnet net;
    alexnet_malloc_params(&net);
    alexnet_param_init(&net);
    alexnet_train(&net, EPOCHS);
    alexnet_free_params(&net);
    return 0;
}