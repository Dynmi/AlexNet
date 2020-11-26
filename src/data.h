#include <stdlib.h>
#include "alexnet.h"

typedef struct{
    float x[IN_CHANNELS][FEATURE0_L][FEATURE0_L];
    float y[OUT_LAYER];
} data;


char* get_random_path();

data load_data_from_file(char **PATH);
void get_random_batch(int n, float *X, float *y, int IMG_SIZE, int LABEL_SIZE);
void get_next_batch(int n, int offset, float *X, float *y, int IMG_SIZE, int LABEL_SIZE);
