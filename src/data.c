#include <stdlib.h>

#include "alexnet.h"
#include "data.h"

void get_random_batch(int n, float *X, float *y, int IMG_SIZE, int LABEL_SIZE)
{
    data d;
    char *path;
    for (int i=0; i<n; i++)
    {
        path = get_random_path();
        d = load_data_from_file(path);
        memcpy(X+i*IMG_SIZE,   d.x, IMG_SIZE*sizeof(float));
        memcpy(y+i*LABEL_SIZE, d.y, LABEL_SIZE*sizeof(float));
    }
}

void get_next_batch(int n, int offset, float *X, float *y, int IMG_SIZE, int LABEL_SIZE)
{
    
}