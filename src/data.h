#include <stdlib.h>

typedef struct{
    int w;
    int h;
    int c;
    float *data;
    int label;
} image;


char** get_random_paths(int label);

image make_image(int w, int h, int c);
image load_image(char *filename, int channels);
void get_random_batch(int n, float *X, float *y, int w, int h, int c, int LABEL_SIZE);
void get_next_batch(int n, int offset, float *X, float *y, int IMG_SIZE, int LABEL_SIZE);
