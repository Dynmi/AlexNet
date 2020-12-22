//
// File:        data.h
// Description: interface of data process functions
// Author:      Haris Wang
//
#include <stdlib.h>

typedef struct{
    int w;
    int h;
    int c;
    float *data;
} image;


char** get_random_paths(int label);

void make_image(image *img, int w, int h, int c);
void free_image(image *img);
image load_image(char *filename, int W, int H, int channels);
image resize_image(image im, int w, int h);

void get_random_batch(int n, float *X, int *Y, 
                        int w, int h, int c, int CLASSES);
void get_next_batch(int n, float *X, float *y, 
                        int offset, int IMG_SIZE, int LABEL_SIZE);
