//
// File:        data.c
// Description: Provide functions for data process 
// Author:      Haris Wang
//

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "data.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


image make_image(int w, int h, int c)
{
    image img;
    img.w = w;
    img.h = h;
    img.c = c;
    img.data = (float *)malloc(h*w*c * sizeof(float));
    return img;
}


void free_image(image *img)
{
    if(img->data){
        free(img->data);
    }
}


static float get_pixel(image img, unsigned int x, unsigned int y, unsigned int c)
{
    assert(x<img.w && y<img.h && c<img.c);
    return img.data[c*img.w*img.h+y*img.w+x];
}


static void set_pixel(image *img, float value, unsigned int x, unsigned int y, unsigned int c)
{
    if (x < 0 || y < 0 || c < 0 || x >= (img->w) || y >= (img->h) || c >= (img->c)) return;
    assert(x<(img->w) && y<(img->h) && c<(img->c));
    img->data[c*(img->w)*(img->h)+y*(img->w)+x] = value;
}


static void add_pixel(image *img, float value, unsigned int x, unsigned int y, unsigned int c)
{
    assert(x<(img->w) && y<(img->h) && c<(img->c));
    img->data[c*(img->w)*(img->h)+y*(img->w)+x] += value;
}


image resize_image(image im, int w, int h)
{
    image   resized = make_image(w, h, im.c), 
            part    = make_image(w, im.h, im.c);
    float   w_scale = (im.w-1) * 1.0 / (w-1),
            h_scale = (im.h-1) * 1.0 / (h-1);
    float   val;
    unsigned int r, c, k;
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(&part, val, c, r, k);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(&resized, val, c, r, k);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(&resized, val, c, r, k);
            }
        }
    }

    free_image(&part);
    return resized;
}


image load_image(char *filename, int W, int H, int channels)
{
    /**
     * load image from file
     * Input:
     *      filename
     *      channels
     * Output:
     * Return:
     *      image
     * */
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if(!data)
    {
        printf("Error! Can't load image %s! \n", filename);
        exit(0);
    }
    if(channels) c=channels;
    image img = make_image(w, h, c);
    for(int k=0; k<c; k++)
    {
        for(int j=0; j<h; j++)
        {
            for(int i=0; i<w; i++)
            {
                int dst_idx = i + w*j + w*h*k;
                int src_idx = k + c*i + c*w*j;
                img.data[dst_idx] = (float)data[src_idx]/255.0;
            }
        }
    }
    free(data);

    if((h&&w) && (H!=img.h || W!=img.w))
    {
        image resized = resize_image(img, H, W);
        free_image(&img);
        return resized;
    }
    return img;
}


void get_random_batch(int n, float *X, float *Y, 
                        int w, int h, int c, int CLASSES)
{
    /**
     * sample random batch of data
     * Input:
     *      n
     *      w, h, c
     *      CLASSES
     * Output:
     *      X
     *      Y
     * */
    image img; 
    img = make_image(w, h, c);
    char path[256];
    int label = 1;
    for (int i=0; i<n; i++)
    {
        sprintf(path, "../dataset/%d/ISIC_000000%d.png", label, rand()%10);
        img = load_image(path, w, h, c);
        memcpy(X+i*w*h*c, img.data, w*h*c*sizeof(float));
        Y[i*CLASSES+label]=1;
    }
}


void get_next_batch(int n, int offset, float *X, float *y, int IMG_SIZE, int LABEL_SIZE)
{

}