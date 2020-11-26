#include <stdlib.h>
#include "alexnet.h"

typedef struct{
    float x[IN_CHANNELS][FEATURE0_L][FEATURE0_L];
    float y[1000];
} data;


data load_data_from_file(char **PATH, char **label);
void get_random_batch(data d, int n, float *X, float *y);
void get_next_batch(data d, int n, int offset, float *X, float *y);
