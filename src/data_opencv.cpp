#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "alexnet.h"
#include "data.h"


data load_data_from_file(char *PATH, int label)
{
    data d;
    Mat img;
    img = imread(PATH, IMREAD_UNCHANGED);
    if (img.empty()) {
        cout << "Error! Input image cannot be read...\n";
        return -1;
    }

    for(int i=0; i<FEATURE0_L; i++)
    {
        for(int j=0; j<FEATURE0_L; j++)
        {
            d.x[0] = img.at<Vec3b>(i, j)[0];
            d.x[1] = img.at<Vec3b>(i, j)[1];
            d.x[2] = img.at<Vec3b>(i, j)[2];
        }
    }

    d.y = label;

    return d;
}


void get_random_batch(int n, float *X, float *y, int IMG_SIZE, int LABEL_SIZE)
{
    /**
     * sample random batch of data
     * Input:
     *      n
     *      IMG_SIZE
     *      LABEL_SIZE
     * Output:
     *      X
     *      y
     * */
    data d;
    char **path = get_random_paths(n);
    for (int i=0; i<n; i++)
    {
        d = load_data_from_file(path[0]);
        memcpy(X+i*IMG_SIZE,   d.x, IMG_SIZE*sizeof(float));
        memcpy(y+i*LABEL_SIZE, d.y, LABEL_SIZE*sizeof(float));
    }
}


void get_next_batch(int n, int offset, float *X, float *y, int IMG_SIZE, int LABEL_SIZE)
{

}