/**
 * 
 * Haris Wang       2020.9.19
 * 
 * */

#include <stdlib.h>
#include <math.h>
#include "alexnet.h"

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define ReLU_BACKWARD(x) x > 0
#define EPSILON 0.0001



static float _CONV(float *x, float *y, int n, int img_width)
{
    /**
     *  x   beggining pointer of input
     *  y   beginning pointer of filter
     *  n   the width of filter
     **/
    float res = 0;
    int x_shift = 0;
    for (int p = 0; p < n * n; p++)
    {
        if (p % n == n - 1)
        {
            x_shift += img_width;
        }
        res += (*(x + x_shift)) * (*(y + p));
        x_shift++;
    }
    return res;
}


void nonlinear_forward(float *x, int units)
{
    for (int i = 0; i < units; i++)
    {
        x[i] = x[i]>0?x[i]:0;
    }
}


void nonlinear_backward(float *x, int units)
{
    for (int i = 0; i < units; i++)
    {
        *(x + i) = ReLU_BACKWARD(*(x + i));
    }
}


void conv_forward(float *input, float *weights, float *bias, float *output, 
                int in_channels, int out_channels, int kernel_size, int padding, int strides, int w, int h)
{
    /**
     * Conv2D forward
     * */

    int out_w, out_h;
    int cur_w, cur_h;
    for(int p=0; p<BATCH_SIZE; p++)
    {
        for (int x = 0 - padding; x < w + padding; x += strides)
        {
            for (int y = 0 - padding; y < h + padding; y += strides)
            {
                for (int out_c = 0; out_c < out_channels; out_c++)
                {
                    // compute output[out_c][out_w][out_h]
                    for (int c = 0; c < in_channels; c++)
                    {
                        /**
                         *  | -------------------------------------------------------------------|
                         *  | input[c][x][y]              input[c][x+kernel_size][y]             |
                         *  | input[c][x][y+kernel_size]  input[c][x+kernel_size][y+kernel_size] |
                         *  | -------------------------------------------------------------------|
                         * 
                         *          conv
                         * 
                         *   weights[out_c][c]
                         * 
                         *          ||
                         *          V
                         * 
                         *   output[c][x][y]
                         * */
                        output[p*out_channels*out_w*out_h + out_c * out_w * out_h + out_w * cur_h] += 
                                    _CONV(input + p*in_channels*w*h + c * w * h + x * h + y, weights + out_c * out_channels + c, kernel_size, w);
                    }
                    output[p*out_channels*out_w*out_h + out_c * out_w * out_h + out_w * cur_h] += bias[out_c];
                }
            }
        }
    }

}


void conv_backward(float *in_error, float *out_error, float *input, float *weights,
                   float *w_deltas, float *b_deltas, int in_channels, int out_channels,
                   int w, int h, int padding, int kernel_size, int strides)
{
    /**
     * Conv2D backward
     * */

    int out_w = (w + 2 * padding - kernel_size) / strides,
        out_h = (h + 2 * padding - kernel_size) / strides;

    for (int i = 0; i < out_w * out_h * out_channels; i++)
    {
        *(b_deltas + i) = *(out_error + i);
    }

    short w_shift = 0,
          in_shift = 0,
          out_shift = 0;

    for (int in_c = 0; in_c < in_channels; in_c++)
    {
        for (int out_c = 0; out_c < out_channels; out_c++)
        {
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    // compute w_deltas
                    for (int out_x = 0; out_x < out_w; out_x++)
                    {
                        for (int out_y = 0; out_y < out_h; out_y++)
                        {
                            for(int p=0; p<BATCH_SIZE; p++)
                            {
                                if (strides * i + out_x - padding<0 | strides * i + out_x - padding> w | strides * j + out_y - padding<0 | strides * j + out_y - padding> h)
                                {
                                    continue;
                                }

                                w_shift = out_c * in_channels * kernel_size * kernel_size + in_c * kernel_size * kernel_size + i * kernel_size + j;
                                in_shift = p * in_channels * w * h + in_c * w * h + (strides * i + out_x - padding) * h + strides * j + out_y - padding;
                                out_shift = out_c * out_w * out_h + out_x * out_h + out_y;
                                w_deltas[w_shift] += input[in_shift] * out_error[out_shift];
                                // w_deltas[out_c][in_c][i][j] += input[p][in_c, strides*i+out_x-padding, strides*j+out_y-padding] * out_error[out_c, out_x, out_y]
                            }
                        }
                    }

                    // compute in_error
                    for (int in_x = 0; in_x < w; in_x++)
                    {
                        for (int in_y = 0; in_y < h; in_y++)
                        {
                            in_shift = in_c * w * h + in_x * h + in_y;
                            out_shift = out_c * out_w * out_h + (in_x + padding + kernel_size - i - 1) * out_h + in_y + padding + kernel_size - j - 1;
                            w_shift = out_c * in_channels * kernel_size * kernel_size + in_c * kernel_size * kernel_size + (kernel_size - i - 1) * kernel_size + kernel_size - j - 1;
                            in_error[in_shift] += out_error[out_shift] * weights[w_shift];
                        }
                    }
                    // in_error[in_c][in_x][in_y] += out_error[out_c][in_x+padding+kernel_size-i-1][in_y+padding+kernel_size-j-1] * weights[out_c][in_c][kernel_size-i-1][kernel_size-j-1]
                }
            }
        }
    }
}


void max_pooling_forward(float *input, float *output, int channels, int in_length, int strides, int pool_size)
{
    /**
     * max pooling forward for multi-channel image
     * */

    int o_x, o_y, o_length = in_length / strides;
    float pixel;

    for(int p=0; p<BATCH_SIZE; p++)
    {
        for(int c = 0; c < channels; c++)
        {
            o_x=0;
            for(int i = 0; i < in_length-strides+1; i += strides)
            {
                o_y=0;
                for(int j = 0; j < in_length-strides+1; j += strides)
                {
                    /**
                     * inputs[i ~ i+pool_size][j ~ j+pool_size]
                     * outputs[o_x][o_j]
                     * */

                    pixel = input[p*channels*in_length*in_length+c*in_length*in_length+i*in_length+j];
                    for(int fx=i; fx<MIN(i+pool_size,in_length); fx++)
                    {
                        for(int fy=j; fy<MIN(j+pool_size,in_length); fy++)
                        {                        
                            pixel = MAX(pixel, input[p*channels*in_length*in_length+c*in_length*in_length+fx*in_length+fy]);
                        }
                    }
                    output[p*channels*o_length*o_length + c*o_length*o_length + o_x + o_y*o_length] = pixel;
                    o_y++;
                }
                o_x++;
            }
        }
    }

}


void max_pooling_backward(int channels, int pool_size, int in_length, float *in_error, float *out_error)
{
    /**
     * max pooling backward for multi-channel image
     * */

    int out_length = in_length / pool_size;

    for (int i = 0; i < out_length; i++)
    {
        for (int j = 0; j < out_length; j++)
        {
            for (int x = i * pool_size; x < MIN((i + 1) * pool_size, in_length); i++)
            {
                for (int y = j * pool_size; y < MIN((j + 1) * pool_size, in_length); y++)
                {
                    *(in_error + channels * in_length * in_length + y * in_length + x) = *(out_error + j * out_length + i);
                }
            }
        }
    }
}


void fc_forward(float *input, float *out, float *weights, float *bias, int in_units, int out_units)
{
    /**
     * fully connected layer forward
     * 
     * 
     * input    (BATCH_SIZE, in_units)
     * out      (BATCH_SIZE, out_units)
     * weights  (in_units, out_units)
     * bias     (out_units)
     * */

    for(int p=0; p<BATCH_SIZE; p++)
    {
        for (int i = 0; i < in_units; i++)
        {
            for (int j = 0; j < out_units; j++)
            {
                out[p*out_units + j] += input[p*in_units + i] * weights[i * out_units + j];
            }
        }
        for (int i = 0; i < out_units; i++)
        {
            out[p*out_units + i] += bias[i];
        }
    }

}


void fc_backward(float *input, float *weights, float *in_error, float *out_error,
                 float *w_deltas, float *b_deltas, int in_units, int out_units)
{
    /**
     * fully connected layer backward
     *
     *      IN
     * input
     * weights
     * out_error
     * out_units
     * 
     *      OUT
     * in_error
     * w_deltas
     * b_deltas
     * */

    for(int p=0; p<BATCH_SIZE; p++)
    {
        for (int i = 0; i < in_units; i++)
        {
            for (int j = 0; j < out_units; j++)
            {
                in_error[i] = weights[i*out_units + j] * out_error[j];
                for (int r = 0; r < BATCH_SIZE; r++)
                {
                    w_deltas[i*out_units + j] += input[p*in_units+i] * out_error[j];
                }
            }
        }
    }

    for (int p = 0; p < out_units; p++)
    {
        b_deltas[p] += out_error[p];
    }
}


void batch_normalization_forward(float *input, float *output, float gamma, float beta, float *avg, int units)
{
    /**
     * batch normalization forward
     * 
     * 
     * input    (BATCH_SIZE, units)
     * output   (BATCH_SIZE, units)
     * */

    float *var = (float *)malloc(sizeof(float) * (units + 1));
    if(var == NULL)
    {
        printf("Can't allocate more memory!!!\n");
        exit(0);
    }
    
    for (int i = 0; i < units; i++)
    {
        avg[i] = 0;
        for (int p = 0; p < BATCH_SIZE; p++)
        {
            avg[i] += input[p * units + i];
        }
        avg[i] /= BATCH_SIZE;
    }

    for (int i = 0; i < units; i++)
    {
        var[i] = 0;
        for (int p = 0; p < BATCH_SIZE; p++)
        {
            var[i] += (input[p*units + i] - avg[i]) * (input[p*units + i] - avg[i]);
        }
        var[i] /= BATCH_SIZE;
    }

    for (int i = 0; i < units; i++)
    {
        for (int p = 0; p < BATCH_SIZE; p++)
        {
            output[p*units + i] = gamma * (input[p*units + i] - avg[p]) / sqrt(var[i] + EPSILON) + beta;
        }
    }

    free(var);
}


void batch_normalization_backward(float *in_error, float *out_error, float delta_gamma,
                                  float delta_beta, float *avg, float gamma, int units)
{
    /**
     * batch normalization backward
     * */

    for (int i = 0; i < units; i++)
    {
        delta_gamma += (*(avg+i)) * (*(out_error + i));
        delta_beta += *(out_error + i);

        *(in_error + i) = (*(out_error + i)) * gamma - (*(avg + i));
    }
}


void softmax_forward(float *input, float *output, int units)
{
    /**
     * softmax layer forward
     * 
     * input    (units)
     * output   (units)
     * */

    float sum;
    for(int p=0; p<BATCH_SIZE; p++)
    {
        sum = 0;
        for (int i = 0; i < units; i++)
        {
            sum += exp(input[p*units+i]);
        }

        for (int i = 0; i < units; i++)
        {
            output[p*units+i] = exp(input[p*units+i]) / sum;
        }
    }

}


void softmax_backward(float *in_error, float *out_error, int units)
{
    /**
     * softmax layer backward
     * */

    for(int i=0; i<units; i++)
    {
        for(int j=0; j<units; j++)
        {
            if(i==j){
                in_error[j] += out_error[j] * (1-out_error[i]);
            }else{
                in_error[j] -= out_error[j] * out_error[i];
            }
        }
    }

    for(int i=0; i<units; i++)
    {
        in_error[i] /= units;
    }
}


void zero_grads(Alexnet *grads)
{
    memset(grads->C1_weights, 0, 4);
    memset(grads->C2_weights, 0, 4);
    memset(grads->C3_weights, 0, 4);
    memset(grads->C4_weights, 0, 4);
    memset(grads->C5_weights, 0, 4);
    memset(grads->FC6weights, 0, 4);
    memset(grads->FC7weights, 0, 4);
    memset(grads->OUTweights, 0, 4);

    memset(grads->C1_bias, 0, 4);
    memset(grads->C2_bias, 0, 4);
    memset(grads->C3_bias, 0, 4);
    memset(grads->C4_bias, 0, 4);
    memset(grads->C5_bias, 0, 4);
    memset(grads->FC6bias, 0, 4);
    memset(grads->FC7bias, 0, 4);

    grads->BN1_gamma = 0;
    grads->BN2_gamma = 0;
    grads->BN3_gamma = 0;
    grads->BN4_gamma = 0;
    grads->BN5_gamma = 0;
    grads->BN1_b = 0;
    grads->BN2_b = 0;
    grads->BN3_b = 0;
    grads->BN4_b = 0;
    grads->BN5_b = 0;
}


void zero_feats(Feature *feats)
{
    memset(feats->input, 0, 4);
    memset(feats->C1, 0, 4);
    memset(feats->BN1, 0, 4);
    memset(feats->P1, 0, 4);

    memset(feats->C2, 0, 4);
    memset(feats->BN2, 0, 4);
    memset(feats->P2, 0, 4);

    memset(feats->C3, 0, 4);
    memset(feats->BN3, 0, 4);

    memset(feats->C4, 0, 4);
    memset(feats->BN4, 0, 4);

    memset(feats->C5, 0, 4);
    memset(feats->BN5, 0, 4);
    memset(feats->P5, 0, 4);

    memset(feats->FC6, 0, 4);

    memset(feats->FC7, 0, 4);

    memset(feats->output, 0, 4);
}


void net_forward(Alexnet *alexnet, Feature *feats)
{
    conv_forward(feats->input, alexnet->C1_weights, alexnet->C1_bias, feats->C1, IN_CHANNELS, C1_CHANNELS, C1_KERNEL_L, 4, C1_STRIDES, FEATURE0_L, FEATURE0_L);
    batch_normalization_forward(feats->C1, feats->BN1, alexnet->BN1_gamma, alexnet->BN1_b, alexnet->BN1_avg, C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    nonlinear_forward(feats->BN1, C1_CHANNELS*FEATURE1_L*FEATURE1_L);

    max_pooling_forward(feats->BN1, feats->P1, C1_CHANNELS, FEATURE1_L, 2, 3);

    conv_forward(feats->P1, alexnet->C2_weights, alexnet->C2_bias, feats->C2, C1_CHANNELS, C2_CHANNELS, C2_KERNEL_L, 1, C2_STRIDES, FEATURE1_L, FEATURE1_L);
    batch_normalization_forward(feats->C2, feats->BN2, alexnet->BN2_gamma, alexnet->BN2_b, alexnet->BN2_avg, C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    nonlinear_forward(feats->BN2, C2_CHANNELS*FEATURE2_L*FEATURE2_L);

    max_pooling_forward(feats->BN2, feats->P2, C2_CHANNELS, FEATURE2_L, 2, 3);

    conv_forward(feats->P2, alexnet->C3_weights, alexnet->C3_bias, feats->C3, C2_CHANNELS, C3_CHANNELS, C3_KERNEL_L, 1, C3_STRIDES, FEATURE2_L, FEATURE2_L);
    batch_normalization_forward(feats->C3, feats->BN3, alexnet->BN3_gamma, alexnet->BN3_b, alexnet->BN3_avg, C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    nonlinear_forward(feats->BN3, C3_CHANNELS*FEATURE3_L*FEATURE3_L);

    conv_forward(feats->BN3, alexnet->C4_weights, alexnet->C4_bias, feats->C4, C3_CHANNELS, C4_CHANNELS, C4_KERNEL_L, 1, C4_STRIDES, FEATURE3_L, FEATURE4_L);
    batch_normalization_forward(feats->C4, feats->BN4, alexnet->BN4_gamma, alexnet->BN4_b, alexnet->BN4_avg, C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    nonlinear_forward(feats->BN4, C4_CHANNELS*FEATURE4_L*FEATURE4_L);

    conv_forward(feats->BN4, alexnet->C5_weights, alexnet->C5_bias, feats->C5, C4_CHANNELS, C5_CHANNELS, C5_KERNEL_L, 1, C5_STRIDES, FEATURE4_L, FEATURE4_L);
    batch_normalization_forward(feats->C5, feats->BN5, alexnet->BN5_gamma, alexnet->BN5_b, alexnet->BN5_avg, C5_CHANNELS*FEATURE5_L*FEATURE5_L);
    nonlinear_forward(feats->BN5, C5_CHANNELS*FEATURE5_L*FEATURE5_L);

    max_pooling_forward(feats->BN5, feats->P5, C5_CHANNELS, FEATURE5_L, 2, 3);

    fc_forward(feats->P5, feats->FC6, alexnet->FC6weights, alexnet->FC6bias, FC6KERNEL_L*C5_CHANNELS*C5_CHANNELS, FC6_LAYER);
    nonlinear_forward(feats->FC6, FC6_LAYER);

    fc_forward(feats->FC6, feats->FC7, alexnet->FC7weights, alexnet->FC7bias, FC6_LAYER, FC7_LAYER);
    nonlinear_forward(feats->FC7, FC7_LAYER);

    softmax_forward(feats->FC7, feats->output, OUT_LAYER);
}


void params_update(Alexnet *alexnet, Alexnet *deltas, float a)
{
    int i;
    float *p_w, *p_d;

    p_w = &(alexnet->C1_weights);
    p_d = &(deltas->C1_weights); 
    for(i=0; i<C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L; i++)
    {
        p_w[i] -= a * p_d[i];
    }

    p_w = &(alexnet->C2_weights);
    p_d = &(deltas->C2_weights); 
    for(i=0; i<C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L; i++)
    {
        p_w[i] -= a * p_d[i];
    }

    p_w = &(alexnet->C3_weights);
    p_d = &(deltas->C3_weights); 
    for(i=0; i<C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L; i++)
    {
        p_w[i] -= a * p_d[i];
    }

    p_w = &(alexnet->C4_weights);
    p_d = &(deltas->C4_weights); 
    for(i=0; i<C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L; i++)
    {
        p_w[i] -= a * p_d[i];
    }

    p_w = &(alexnet->C5_weights);
    p_d = &(deltas->C5_weights); 
    for(i=0; i<C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L; i++)
    {
        p_w[i] -= a * p_d[i];
    }

    p_w = &(alexnet->FC6weights);
    p_d = &(deltas->FC6weights); 
    for(i=0; i<C5_CHANNELS*FC6_LAYER*FC6KERNEL_L*FC6KERNEL_L; i++)
    {
        p_w[i] -= a * p_d[i];
    }

    p_w = &(alexnet->FC7weights);
    p_d = &(deltas->FC7weights); 
    for(i=0; i<FC6_LAYER*FC7_LAYER; i++)
    {
        p_w[i] -= a * p_d[i];
    }

    p_w = &(alexnet->OUTweights);
    p_d = &(deltas->OUTweights); 
    for(i=0; i<FC7_LAYER*OUT_LAYER; i++)
    {
        p_w[i] -= a * p_d[i];
    }

    for(i=0; i<C1_CHANNELS; i++)
    {
        alexnet->C1_bias[i] -= a * deltas->C1_bias[i];
    }

    for(i=0; i<C2_CHANNELS; i++)
    {
        alexnet->C2_bias[i] -= a * deltas->C2_bias[i];
    }

    for(i=0; i<C3_CHANNELS; i++)
    {
        alexnet->C3_bias[i] -= a * deltas->C3_bias[i];
    }

    for(i=0; i<C4_CHANNELS; i++)
    {
        alexnet->C4_bias[i] -= a * deltas->C4_bias[i];
    }

    for(i=0; i<C5_CHANNELS; i++)
    {
        alexnet->C5_bias[i] -= a * deltas->C5_bias[i];
    }

    for(i=0; i<FC6_LAYER; i++)
    {
        alexnet->FC6bias[i] -= a * deltas->FC6bias[i];
    }

    for(i=0; i<FC7_LAYER; i++)
    {
        alexnet->FC7bias[i] -= a * deltas->FC7bias[i];
    }

    alexnet->BN1_gamma -= a * deltas->BN1_gamma;
    alexnet->BN2_gamma -= a * deltas->BN2_gamma; 
    alexnet->BN3_gamma -= a * deltas->BN3_gamma; 
    alexnet->BN4_gamma -= a * deltas->BN4_gamma; 
    alexnet->BN5_gamma -= a * deltas->BN5_gamma; 
    alexnet->BN1_b -= a * deltas->BN1_b; 
    alexnet->BN2_b -= a * deltas->BN2_b; 
    alexnet->BN3_b -= a * deltas->BN3_b; 
    alexnet->BN4_b -= a * deltas->BN4_b; 
    alexnet->BN5_b -= a * deltas->BN5_b; 
}


void net_backward(Feature *error, Alexnet *alexnet, Alexnet *deltas, Feature *feats)
{

    softmax_backward(error->output, error->FC7, OUT_LAYER);

    nonlinear_backward(error->FC7, FC7_LAYER);
    fc_backward(feats->FC7, alexnet->FC7weights, error->FC6, error->FC7, deltas->FC7weights, deltas->FC7bias, FC6_LAYER, FC7_LAYER);

    nonlinear_backward(error->FC6, FC6_LAYER);
    fc_backward(feats->FC6, alexnet->FC6weights, error->P5, error->FC6, deltas->FC6weights, deltas->FC6bias, FC6KERNEL_L*C5_CHANNELS*C5_CHANNELS, FC6_LAYER);

    max_pooling_backward(C5_CHANNELS, 3, FEATURE5_L, error->BN5, error->P5);

    nonlinear_backward(error->BN5, C5_CHANNELS*FEATURE5_L*FEATURE5_L);
    batch_normalization_backward(error->C5, error->BN5, deltas->BN5_gamma, deltas->BN5_b, alexnet->BN5_avg, alexnet->BN5_gamma, C5_CHANNELS*FEATURE5_L*FEATURE5_L);
    conv_backward(error->BN4, error->C5, feats->C5, alexnet->C5_weights, deltas->C5_weights, deltas->C5_bias, 
                    C4_CHANNELS, C5_CHANNELS, FEATURE5_L, FEATURE5_L, 1, C5_KERNEL_L, 1);

    nonlinear_backward(error->BN4, C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    batch_normalization_backward(error->C4, error->BN4, deltas->BN4_gamma, deltas->BN4_b, alexnet->BN4_avg, alexnet->BN4_gamma, C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    conv_backward(error->BN3, error->C4, feats->C4, alexnet->C4_weights, deltas->C4_weights, deltas->C4_bias, 
                    C3_CHANNELS, C4_CHANNELS, FEATURE4_L, FEATURE4_L, 1, C4_KERNEL_L, 1);

    nonlinear_backward(error->BN3, C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    batch_normalization_backward(error->C3, error->BN3, deltas->BN3_gamma, deltas->BN3_b, alexnet->BN3_avg, alexnet->BN3_gamma, C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    conv_backward(error->P2, error->C3, feats->C3, alexnet->C3_weights, deltas->C3_weights, deltas->C3_bias, 
                    C2_CHANNELS, C3_CHANNELS, FEATURE3_L, FEATURE3_L, 1, C3_KERNEL_L, 1);

    max_pooling_backward(C2_CHANNELS, 3, FEATURE2_L, error->BN2, error->P2);

    nonlinear_backward(error->BN2, C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    batch_normalization_backward(error->C2, error->BN2, deltas->BN2_gamma, deltas->BN2_b, alexnet->BN2_avg, alexnet->BN2_gamma, C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    conv_backward(error->P1, error->C2, feats->C2, alexnet->C2_weights, deltas->C2_weights, deltas->C2_bias, 
                    C1_CHANNELS, C2_CHANNELS, FEATURE2_L, FEATURE2_L, 1, C2_KERNEL_L, 1);

    max_pooling_backward(C1_CHANNELS, 3, FEATURE1_L, error->BN1, error->P1);

    nonlinear_backward(error->BN1, C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    batch_normalization_backward(error->C1, error->BN1, deltas->BN1_gamma, deltas->BN1_b, alexnet->BN1_avg, alexnet->BN1_gamma, C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    conv_backward(error->input, error->C1, feats->C1, alexnet->C1_weights, deltas->C1_weights, deltas->C1_bias, 
                    IN_CHANNELS, C1_CHANNELS, FEATURE1_L, FEATURE1_L, 4, C1_KERNEL_L, 0);


    params_update(alexnet, deltas, 0.001);

}


void compute_mse_error(float *error, float *preds, float *labels, int units)
{
    /**
     * Compute error between 'preds' and 'labels', then send to 'error'
     * 
     * error    [units]
     * preds    [BATCH_SIZE, units]
     * labels   [BATCH_SIZE, units]
     * */

    for(int i=0; i<units; i++)
    {
        for(int p=0; p<BATCH_SIZE; p++)      
        {
            error[i] += (preds[p*units+i]-labels[p*units+i]) * (preds[p*units+i]-labels[p*units+i]);
        }
        error[i] /= BATCH_SIZE;
    }

}


void train(Alexnet *alexnet)
{
    Feature feats;
    float *batch_x = (float *)malloc(4*BATCH_SIZE*3*FEATURE0_L*FEATURE0_L);
    float *batch_y = (float *)malloc(4*BATCH_SIZE*OUT_LAYER);

    /** 
     * sample batch data
     * */

    net_forward(alexnet, &feats);

    Alexnet deltas;
    Feature error;
    zero_feats(&error);

    compute_mse_error(error.output, feats.output, batch_y, OUT_LAYER);

    zero_grads(&deltas);
    net_backward(&error, alexnet, &deltas, &feats);

    free(batch_x);
    free(batch_y);



}

void predict(Alexnet *alexnet, float *inputs, float *outputs)
{

    Feature feats;
    memcpy(feats.input, inputs, sizeof(feats.input));
    net_forward(alexnet, &feats);
    memcpy(outputs, feats.output, sizeof(feats.output));
    
}