//
// File:        alexnet.c
// Description: Implemention of alexnet-related operations
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "alexnet.h"
#include "hyperparams.h"

struct timespec start, finish; float duration;


#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define EPSILON 0.00001


typedef struct conv_args{
    conv_op *op;
    int batch_id;
    pthread_mutex_t *mtx;
} conv_args;

void pthread_conv_op_forward(void *argv)
{
    /**
     * pthread conv_op_forward
     * */
    conv_args cp;
    memcpy(&cp, (conv_args *)argv, sizeof(conv_args));
    float *input = cp.op->input;
    float *weights = cp.op->weights;
    float *bias = cp.op->bias;
    float *output = cp.op->output;
    int in_channels = cp.op->in_channels; int out_channels = cp.op->out_channels; 
    int kernel_size = cp.op->kernel_size; int padding=cp.op->padding; 
    int strides = cp.op->stride; int w = cp.op->in_w; int h = cp.op->in_h;
    int p = cp.batch_id;

    int out_w, out_h;
    out_w = cp.op->out_w; out_h = cp.op->out_h;

    unsigned int input_shift, weights_shift, out_shift;
    int cur_w, cur_h;
    for (int out_c = 0; out_c < out_channels; out_c++)
    {
        cur_w = 0; 
        for (int x = 0 - padding; x < w + padding-kernel_size+1; x += strides)
        {
            cur_h = 0;
            for (int y = 0 - padding; y < h + padding-kernel_size+1; y += strides)
            {
                //printf("cur_w is %d;  cur_h is %d\n", cur_w, cur_h);
            
                // output[out_c][cur_w][cur_h]
                out_shift = p*out_channels*out_w*out_h + out_c*out_w*out_h + cur_w*out_h + cur_h;
                //cp.output[out_shift] = 0.0;
                for (int in_c = 0; in_c < in_channels; in_c++)
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
                     *   output[c][cur_w][cur_h]
                     * */
                    float tmp = 0;

                    for(short kernel_x=0; kernel_x<kernel_size; kernel_x++)
                    {
                        if(x+kernel_x<0) continue; // padding areas
                        if(x+kernel_x>=w) break;

                        for(short kernel_y=0; kernel_y<kernel_size; kernel_y++)
                        {
                            if(y+kernel_y<0) continue; // padding areas
                            if(y+kernel_y>=w) break;

                            // res += input[p][in_c][x][y] * weights[out_c][in_c][i][j]
                            input_shift = p*cp.op->in_units + in_c*w*h + (x+kernel_x)*w + (y+kernel_y);
                            weights_shift = out_c*in_channels*kernel_size*kernel_size + in_c*kernel_size*kernel_size + kernel_x*kernel_size + kernel_y;
                            tmp += input[input_shift] * weights[weights_shift];
                        }
                    }
                    
                    output[out_shift] += tmp;
                }
                output[out_shift] += bias[out_c];
                // printf("%.2f \n", x, y, output[p*out_channels*out_w*out_h + out_c*out_w*out_h + cur_w*out_h + cur_h]);
                cur_h++;
            }
            cur_w++;
        }
    }    
}

void conv_op_forward(conv_op *op)
{
    /**
     * conv2d forward
     * 
     * Input
     *      op->input
     *      op->weights
     *      op->bias
     * Output
     *      op->output
     * */
    conv_args args[BATCH_SIZE+1];
    pthread_t tid[BATCH_SIZE+1];
    for(int p=0; p<BATCH_SIZE; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        pthread_create(&tid[p], NULL, pthread_conv_op_forward, (void *)(&args[p]));
    }
    for(int p=0; p<BATCH_SIZE; p++)
    {
        pthread_join(tid[p], NULL);
    }

}


void pthread_conv_op_backward(void *argv)
{
    /**
     * pthread conv_op_backward
     * */
    conv_args cp;
    memcpy(&cp, (conv_args *)argv, sizeof(conv_args));

    float *in_error = cp.op->d_input;
    float *out_error = cp.op->d_output;
    float *w_deltas = cp.op->d_weights;
    float *b_deltas = cp.op->d_bias;
    float *input = cp.op->input;
    float *weights = cp.op->weights;
    int in_channels = cp.op->in_channels; 
    int out_channels = cp.op->out_channels;
    int in_w = cp.op->in_w, in_h = cp.op->in_h;
    int out_w = cp.op->out_w, out_h = cp.op->out_h;
    int padding = cp.op->padding;
    int kernel_size = cp.op->kernel_size;
    int strides = cp.op->stride;

    int p = cp.batch_id;

    if(p==0)
    {
        // compute b_deltas
        for (int c=0; c<out_channels; c++)
        {
            for (int i=0; i<out_w*out_h; i++)
            {
                b_deltas[c] += out_error[c*out_w*out_h+i];
            }
        }
    }

    unsigned int w_shift, in_shift, out_shift;
    for (int in_c = 0; in_c < in_channels; in_c++)
    {
        for (int out_c = 0; out_c < out_channels; out_c++)
        {
            for (int out_x = 0; out_x < out_w; out_x++)
            {
                for (int out_y = 0; out_y < out_h; out_y++)
                {
                    for (int i = 0; i < kernel_size; i++)
                    {
                        if (strides*out_x+i-padding < 0 | strides*out_x+i-padding >= in_w)
                            continue;

                        for (int j = 0; j < kernel_size; j++)
                        {
                            if (strides*out_y+j-padding < 0 | strides*out_y+j-padding >= in_h)
                                continue;
                            
                            out_shift = out_c*out_w*out_h + out_x*out_h + out_y;
                            w_shift = out_c*in_channels*kernel_size*kernel_size + in_c*kernel_size*kernel_size + i*kernel_size + j;

                            // compute w_deltas[out_c][in_c][i][j]
                            
                            pthread_mutex_lock( cp.mtx+w_shift );
                            in_shift = p*in_channels*in_w*in_h + in_c*in_w*in_h + (strides*out_x+i-padding)*in_h + (strides*out_y+j-padding);
                            w_deltas[w_shift] += input[in_shift] * out_error[out_shift] / BATCH_SIZE;
                            pthread_mutex_unlock( cp.mtx+w_shift );

                            if(p==0)
                            {
                                // compute in_error[in_c][][]
                                in_shift = in_c*in_w*in_h + (strides*out_x+i-padding)*in_h + (strides*out_y+j-padding);
                                w_shift = out_c*in_channels*kernel_size*kernel_size + in_c*kernel_size*kernel_size + (kernel_size-i-1)*kernel_size + j;                                
                                in_error[in_shift] += out_error[out_shift] * weights[w_shift];
                            }
                        }
                    }
                }
            }
        }
    }

}

void conv_op_backward(conv_op *op)
{
    /**
     * conv2d backward
     * 
     * Input
     *      op->d_output
     * Output
     *      op->d_weights
     *      op->d_bias
     *      op->d_input
     * */
    pthread_mutex_t *w_deltas_mtx = (pthread_mutex_t *)malloc(op->in_channels * op->out_channels * op->kernel_size * op->kernel_size * sizeof(pthread_mutex_t));
    for(int i=0; i< op->in_channels * op->out_channels * op->kernel_size * op->kernel_size; i++)
    {
        pthread_mutex_init(w_deltas_mtx+i, NULL);
    }

    conv_args args[BATCH_SIZE+1];
    pthread_t tid[BATCH_SIZE+1];
    for(int p=0; p<BATCH_SIZE; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        args[p].mtx = w_deltas_mtx;
        pthread_create(&tid[p], NULL, pthread_conv_op_backward, (void *)(&args[p]));
    }
    for(int p=0; p<BATCH_SIZE; p++)
    {
        pthread_join(tid[p], NULL);
    }

    free(w_deltas_mtx);
}


void max_pooling_op_forward(max_pooling_op *op)
{
    float *input = op->input;
    float *output = op->output;
    int channels = op->channels;
    int in_length = op->in_w;
    int strides = op->stride;
    int pool_size = op->kernel_size;

    int o_x, o_y, o_length = in_length / strides;
    float pixel;

    for (int p=0; p<BATCH_SIZE; p++)
    {
        for (int c = 0; c < channels; c++)
        {
            o_x=0;
            for (int i = 0; i < in_length-strides+1; i += strides)
            {
                o_y=0;
                for (int j = 0; j < in_length-strides+1; j += strides)
                {
                    /**
                     * inputs[i ~ i+pool_size][j ~ j+pool_size]
                     * outputs[o_x][o_j]
                     * */

                    pixel = input[p*channels*in_length*in_length+c*in_length*in_length+i*in_length+j];
                    for (int fx=i; fx<MIN(i+pool_size,in_length); fx++)
                    {
                        for (int fy=j; fy<MIN(j+pool_size,in_length); fy++)
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


void max_pooling_op_backward(max_pooling_op *op)
{
    float *in_error = op->d_input;
    float *out_error = op->d_output;
    float *input = op->input;
    int channels = op->channels;
    int pool_size = op->kernel_size;

    int in_w = op->in_w; int in_h = op->in_h;
    int out_w = op->out_w; int out_h = op->out_h;
    
    int in_x, in_y;
    float max_value, cur_value;
    int x, y;
    int in_shift, out_shift;
    for (int c=0; c<channels; c++)
    {
        for (int i=0; i < op->out_w; i++)
        {
            for (int j=0; j < op->out_h; j++)
            {
                for (int p=0; p<BATCH_SIZE; p++)
                {
                    //
                    // output[p][c][i][j]
                    //
                    x = i*pool_size;    
                    y = j*pool_size;
                    max_value = -1111111;
                    while ( x<MIN((i + 1) * pool_size, in_w) )
                    {
                        while ( y<MIN((j + 1) * pool_size, in_h) )
                        {
                            cur_value = input[p*channels*in_w*in_h + c*in_w*in_h + y*in_w + x];
                            if(cur_value>max_value)
                            {
                                max_value = cur_value;
                                in_x = x;
                                in_y = y;
                            }
                            y++;
                        }
                        x++;
                    }
                    
                    in_shift = c*in_w*in_h + in_y*in_w + in_x;
                    out_shift = c*out_w*out_h + j*out_w + i;
                    in_error[in_shift] += out_error[out_shift] / BATCH_SIZE;
                }
            }
        }
    }

}


typedef struct fc_args{
    fc_op *op;
    int batch_id;
    pthread_mutex_t *mtx;
} fc_args;

void pthread_fc_op_forward(void *argv)
{
    /**
     * pthread fc_op_forward
     * */
    fc_args args;
    memcpy(&args, (fc_args *)argv, sizeof(fc_args));

    float *input = args.op->input;
    float *output = args.op->output;
    float *weights = args.op->weights;
    float *bias = args.op->bias;
    int in_units = args.op->in_units;
    int out_units = args.op->out_units;
    int p = args.batch_id;

    for (int i = 0; i < in_units; i++)
    {
        for (int j = 0; j < out_units; j++)
        {
            output[p*out_units + j] += input[p*in_units + i] * weights[j*out_units + i];
        }
    }
    for (int i = 0; i < out_units; i++)
    {
        output[p*out_units + i] += bias[i];
    }

}

void fc_op_forward(fc_op *op)
{
    fc_args fp[BATCH_SIZE];
    pthread_t tid[BATCH_SIZE+1];
    for(int p=0; p<BATCH_SIZE; p++)
    {
        fp[p].op = op;
        fp[p].batch_id = p;
        pthread_create(&tid[p], NULL, pthread_fc_op_forward, (void *)(&fp[p]));
    }
    for(int p=0; p<BATCH_SIZE; p++)
        pthread_join(tid[p], NULL);
}


void pthread_fc_op_backward(void *argv)
{
    /**
     * pthread fc_op_backward
     * */
    fc_args args;
    memcpy(&args, (fc_args *)argv, sizeof(fc_args));

    float *input = args.op->input;
    float *weights = args.op->weights;
    float *bias = args.op->bias;
    int in_units = args.op->in_units;
    int out_units = args.op->out_units;
    float *in_error = args.op->d_input;
    float *out_error = args.op->d_output;
    float *w_deltas = args.op->d_weights;
    float *b_deltas = args.op->d_bias;
    
    int p = args.batch_id;
    unsigned int w_shift;
    for (int i=0; i<in_units; i++)
    {
        for (int j=0; j<out_units; j++)
        {
            w_shift = i*out_units + j;
            if(p==0)
            {
                in_error[i] += weights[w_shift] * out_error[j];
            }

            pthread_mutex_lock( args.mtx+w_shift );
            w_deltas[w_shift] += input[p*in_units + i] * out_error[j] / BATCH_SIZE;
            pthread_mutex_unlock( args.mtx+w_shift );
        }
    }

    if(p==0)
    {
        for (int p = 0; p < out_units; p++)
        {
            b_deltas[p] = out_error[p];
        }
    }

}

void fc_op_backward(fc_op *op)
{
    pthread_mutex_t *w_deltas_mtx = (pthread_mutex_t *)malloc(op->in_units * op->out_units * sizeof(pthread_mutex_t));
    for(int i=0; i< op->in_units * op->out_units; i++)
    {
        pthread_mutex_init(w_deltas_mtx+i, NULL);
    }
    conv_args args[BATCH_SIZE+1];
    pthread_t tid[BATCH_SIZE+1];
    for(int p=0; p<BATCH_SIZE; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        args[p].mtx = w_deltas_mtx;
        pthread_create(&tid[p], NULL, pthread_fc_op_backward, (void *)(&args[p]));
    }
    for(int p=0; p<BATCH_SIZE; p++)
    {
        pthread_join(tid[p], NULL);
    }

    free(w_deltas_mtx);
}


void batch_norm_op_forward(batch_norm_op *op)
{
    float *input = op->input;
    float *output = op->output;
    float *beta = op->beta;
    float *gamma = op->gamma;
    int units = op->units;

    op->avg = (float *)calloc(op->units, sizeof(float));
    op->var = (float *)calloc(op->units, sizeof(float));
    op->x_norm = (float *)malloc(sizeof(float) * BATCH_SIZE * op->units);
    // calculate mean for each unit along batch axis
    for (int i = 0; i < units; i++)
    {
        for (int p = 0; p < BATCH_SIZE; p++)
            op->avg[i] += input[p*units + i];
        op->avg[i] /= BATCH_SIZE;
    }

    // calculate variance for each unit along batch axis
    for (int i = 0; i < units; i++)
    {
        for (int p = 0; p < BATCH_SIZE; p++)
            op->var[i] += (input[p*units + i] - op->avg[i]) * (input[p*units + i] - op->avg[i]);
        op->var[i] /= BATCH_SIZE;
    }

    for (int i = 0; i < units; i++)
    {
        for (int p = 0; p < BATCH_SIZE; p++)
        {
            op->x_norm[p*units+i] = (input[p*units + i] - op->avg[i]) / sqrt(op->var[i] + EPSILON); 
            output[p*units+i] = gamma[i] * op->x_norm[p*units+i] + beta[i];
            //output[p*units+i] = op->x_norm[p*units+i];
        }
    }
}


void batch_norm_op_backward(batch_norm_op *op)
{
    float *in_error = op->d_input;
    float *out_error = op->d_output;
    float *delta_gamma = op->d_gamma;
    float *delta_beta = op->d_beta;
    float *gamma = op->gamma;
    int units = op->units;

    float *x_norm_avg = (float *)calloc(units, sizeof(float));
    float nn = 1.0 / BATCH_SIZE;
    for (int i = 0; i < units; i++)
    {
        for(int p=0; p<BATCH_SIZE; p++)
            delta_gamma[i] += op->x_norm[p*units+i] * out_error[i];
        delta_gamma[i] /= BATCH_SIZE;

        delta_beta[i] += out_error[i];

        for(int p=0; p<BATCH_SIZE; p++)
            x_norm_avg[i] += op->x_norm[p*units+i];
        x_norm_avg[i] /= BATCH_SIZE;
    }

    for (int i = 0; i < units; i++)
    {
        in_error[i] = 0 - gamma[i] * out_error[i] * x_norm_avg[i] / sqrt(op->var[i]+EPSILON) ; 
        //in_error[i] = out_error[i];
    } 
    free(x_norm_avg);
    free(op->avg);
    free(op->var);
    free(op->x_norm);
}


void relu_op_forward(nonlinear_op *op)
{
    for (int p = 0; p < BATCH_SIZE; p++)
    {
        for (int i = 0; i < (op->units); i++)
        {
            op->output[p*(op->units)+i] = op->input[p*(op->units)+i] * (op->input[p*(op->units)+i]>0);
        }
    }
}


void relu_op_backward(nonlinear_op *op)
{
    float tmp;
    for (int i = 0; i < (op->units); i++)
    {
        tmp=0;
        for (int p = 0; p < BATCH_SIZE; p++)
            tmp += (op->input[p*(op->units)+i] > 0);
        tmp /= BATCH_SIZE;

        op->d_input[i] = op->d_output[i] * tmp;
    }
}


void sigmoid_op_forward(nonlinear_op *op)
{
    for (int p=0; p<BATCH_SIZE; p++)
    {
        for (int i = 0; i < (op->units); i++)
        {
            op->output[p*(op->units)+i] = 1.0 / ( 1.0 + exp( 0.0 - op->input[p*(op->units)+i] ) );
        }
    }
}


void sigmoid_op_backward(nonlinear_op *op)
{
    float tmp;
    for (int i = 0; i < (op->units); i++)
    {
        tmp=0;
        for (int p = 0; p < BATCH_SIZE; p++)
            tmp += op->output[p*(op->units)+i] * (1 - op->output[p*(op->units)+i]);
        tmp /= BATCH_SIZE;

        op->d_input[i] = op->d_output[i] * tmp;
    }
}

void softmax_op_forward(nonlinear_op *op)
{
    for(int p=0; p<BATCH_SIZE; p++)
    {
        float esum=0;
        for(int i=0; i< op->units; i++)
            esum += exp( op->input[i + p * op->units]); 

        for(int i=0; i< op->units; i++)
            op->output[i + p * op->units] = exp( op->input[i + p * op->units]) / esum;       
    }
}

void softmax_op_backward(nonlinear_op *op)
{
    for(int i=0; i< op->units; i++)
    {
        for(int j=0; j< op->units; j++)
        {
            if(i==j){
                op->d_input[j] += op->d_output[j] * (1 - op->d_output[i]);
            }else{
                op->d_input[j] -= op->d_output[j] * op->d_output[i];
            }
        }
    }

    for(int i=0; i< op->units; i++)
    {
        op->d_input[i] /= op->units;
    }
}


void cross_entropy_loss(float *delta_preds, const float *preds, const int *labels, int units)
{
    /**
     * Cross Entropy backward
     * 
     * Input
     *      preds   [BATCH_SIZE, units]
     *      labels  [BATCH_SIZE]
     * Output
     *      delta_preds [units]
     * */
    float ce_loss = 0;
    for(int p=0; p<BATCH_SIZE; p++)
    {
        float esum = 0;
        for(int i=0; i<units; i++)
        {
            esum += exp(preds[i+p*units]);
        }

        ce_loss += 0-log(exp(preds[labels[p]+p*units]) / esum);

        for(int i=0; i<units; i++)
        {
            // preds[i+p*units]
            if(labels[p]==i)
            {
                delta_preds[i] += exp(preds[i+p*units]) / esum - 1;
            }else{
                delta_preds[i] += exp(preds[i+p*units]) / esum;
            } 
        }
    }
    ce_loss /= BATCH_SIZE;
    printf("cross entropy loss on batch data is %f \n", ce_loss);

    for(int i=0; i<units; i++)
    {
        delta_preds[i] /= BATCH_SIZE;
        //printf("delta_preds%d  %f \n", i, delta_preds[i]);
    }
}


void dropout(float *x, float prob, int units)
{
    /**
     * dropout regularization
     * 
     * Input:
     *      x   [BATCH_SIZE, units]
     *      prob    prob~(0,1)
     * Output:
     *      x   [BATCH_SIZE, units]
     * */
    for(int p=0; p<BATCH_SIZE; p++)
    {
        for(int i=0; i<units; i++)
        {
            if(rand()%100 < prob*100)
            {
                x[p*units+i] = 0;
            }
        }
    }
}


static float v_conv1_weights[C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L];
static float v_conv2_weights[C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L];
static float v_conv3_weights[C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L];
static float v_conv4_weights[C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L];
static float v_conv5_weights[C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L];
static float v_fc1_weights[C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L];
static float v_fc2_weights[FC6_LAYER*FC7_LAYER];
static float v_fc3_weights[FC7_LAYER*OUT_LAYER];

static float v_conv1_bias[C1_CHANNELS];
static float v_conv2_bias[C2_CHANNELS];
static float v_conv3_bias[C3_CHANNELS];
static float v_conv4_bias[C4_CHANNELS];
static float v_conv5_bias[C5_CHANNELS];
static float v_fc1_bias[FC6_LAYER];
static float v_fc2_bias[FC7_LAYER];
static float v_fc3_bias[OUT_LAYER];

static float v_bn1_gamma[C1_CHANNELS*FEATURE1_L*FEATURE1_L];
static float v_bn2_gamma[C2_CHANNELS*FEATURE2_L*FEATURE2_L];
static float v_bn3_gamma[C3_CHANNELS*FEATURE3_L*FEATURE3_L];
static float v_bn4_gamma[C4_CHANNELS*FEATURE4_L*FEATURE4_L];
static float v_bn5_gamma[C5_CHANNELS*FEATURE5_L*FEATURE5_L];
static float v_bn1_beta[C1_CHANNELS*FEATURE1_L*FEATURE1_L];
static float v_bn2_beta[C2_CHANNELS*FEATURE2_L*FEATURE2_L];
static float v_bn3_beta[C3_CHANNELS*FEATURE3_L*FEATURE3_L];
static float v_bn4_beta[C4_CHANNELS*FEATURE4_L*FEATURE4_L];
static float v_bn5_beta[C5_CHANNELS*FEATURE5_L*FEATURE5_L];


void clip(float *x, float down, float up)
{
    *x = MIN(up, MAX(down, *x));
}

void momentum_sgd(float *w, float *v_w, float *d_w, int units)
{
    /**
     * momentum stochastic gradient descent
     * 
     * Input
     *      w   [units]
     *      v_w [units]
     *      d_w [units]
     * Output
     *      w   [units]
     *      v_w [units]
     * */     
    for(int i=0; i<units; i++)
    {
        v_w[i] = 0.2 * v_w[i] - 0.8 * LEARNING_RATE * d_w[i];
        clip(v_w+i, -1, 1);
        w[i] = w[i] + v_w[i];
    }
}


void update_params(alexnet *net)
{
    momentum_sgd(net->conv1.weights, v_conv1_weights, net->conv1.d_weights, C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L);
    momentum_sgd(net->conv2.weights, v_conv2_weights, net->conv2.d_weights, C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L);
    momentum_sgd(net->conv3.weights, v_conv3_weights, net->conv3.d_weights, C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L);
    momentum_sgd(net->conv4.weights, v_conv4_weights, net->conv4.d_weights, C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L);
    momentum_sgd(net->conv5.weights, v_conv5_weights, net->conv5.d_weights, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L);
    momentum_sgd(net->fc1.weights,   v_fc1_weights,   net->fc1.d_weights,   C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L);
    momentum_sgd(net->fc2.weights,   v_fc2_weights,   net->fc2.d_weights,   FC6_LAYER*FC7_LAYER);
    momentum_sgd(net->fc3.weights,   v_fc3_weights,   net->fc3.d_weights,   FC7_LAYER*OUT_LAYER);

    momentum_sgd(net->conv1.bias,   v_conv1_bias,   net->conv1.d_bias, C1_CHANNELS);
    momentum_sgd(net->conv2.bias,   v_conv2_bias,   net->conv2.d_bias, C2_CHANNELS);
    momentum_sgd(net->conv3.bias,   v_conv3_bias,   net->conv3.d_bias, C3_CHANNELS);
    momentum_sgd(net->conv4.bias,   v_conv4_bias,   net->conv4.d_bias, C4_CHANNELS);
    momentum_sgd(net->conv5.bias,   v_conv5_bias,   net->conv5.d_bias, C5_CHANNELS);
    momentum_sgd(net->fc1.bias,     v_fc1_bias,     net->fc1.d_bias,   FC6_LAYER);
    momentum_sgd(net->fc2.bias,     v_fc2_bias,     net->fc2.d_bias,   FC7_LAYER);
    momentum_sgd(net->fc3.bias,     v_fc3_bias,     net->fc3.d_bias,   OUT_LAYER);

    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);
    momentum_sgd(net->bn1.gamma,     v_bn1_gamma,     net->bn1.d_gamma,   OUT_LAYER);

    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
    momentum_sgd(net->bn1.beta,     v_bn1_beta,     net->bn1.d_beta,   OUT_LAYER);
}


void alexnet_forward(alexnet *net)
{
    net->conv1.input = net->input;

    //printf(">>>>>>>>>>>>>>>>>>conv1>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv1.output = (float *)calloc(BATCH_SIZE * net->conv1.out_units, sizeof(float));
    conv_op_forward(&(net->conv1));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv1) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn1.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->bn1.units);
    net->bn1.input = net->conv1.output;
    batch_norm_op_forward(&(net->bn1));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn1)) duration: %.2fs \n", duration);
#endif

    for(int p=0; p< net->bn1.units * BATCH_SIZE; p++)
    {
        if(net->bn1.output[p]<(0-10) | net->bn1.output[p]>10 )
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: BatchNorm1 error !!!  %f\n", net->bn1.output[p]);
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu1.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->relu1.units);
    net->relu1.input = net->bn1.output;
    relu_op_forward(&(net->relu1));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->relu1)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp1.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->mp1.out_units);
    net->mp1.input = net->relu1.output;
    max_pooling_op_forward(&(net->mp1));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->mp1)) duration: %.2fs \n", duration);
#endif

    //printf(">>>>>>>>>>>>>>>>>>conv2>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv2.output = (float *)calloc(BATCH_SIZE * net->conv2.out_units, sizeof(float));
    net->conv2.input = net->mp1.output;
    conv_op_forward(&(net->conv2));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv2)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn2.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->bn2.units);
    net->bn2.input = net->conv2.output;
    batch_norm_op_forward(&(net->bn2));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn2)) duration: %.2fs \n", duration);
#endif

    for(int p=0; p< net->bn2.units * BATCH_SIZE; p++)
    {
        if(net->bn2.output[p]<(0-10) | net->bn2.output[p]>10 )
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: BatchNorm2 error !!! %f\n", net->bn2.output[p]);
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu2.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->relu2.units);
    net->relu2.input = net->bn2.output;
    relu_op_forward(&(net->relu2));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->relu2)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp2.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->mp2.out_units);
    net->mp2.input = net->relu2.output;
    max_pooling_op_forward(&(net->mp2));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->mp2)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    //printf(">>>>>>>>>>>>>>>>>>conv3>>>>>>>>>>>>>>>>>>>>>>>>> \n");
    net->conv3.output = (float *)calloc(BATCH_SIZE * net->conv3.out_units, sizeof(float));
    net->conv3.input = net->mp2.output;
    conv_op_forward(&(net->conv3));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv3)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn3.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->bn3.units);
    net->bn3.input = net->conv3.output;
    batch_norm_op_forward(&(net->bn3));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn3)); duration: %.2fs \n", duration);
#endif

    for(int p=0; p< net->bn3.units * BATCH_SIZE; p++)
    {
        if(net->bn3.output[p]<(0-10) | net->bn3.output[p]>10 )
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: BatchNorm3 error !!! %f \n", net->bn3.output[p]);
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu3.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->relu3.units);
    net->relu3.input = net->bn3.output;
    relu_op_forward(&(net->relu3));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->relu3)) duration: %.2fs \n", duration);
#endif

    //printf(">>>>>>>>>>>>>>>>>>conv4>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv4.output = (float *)calloc(BATCH_SIZE * net->conv4.out_units, sizeof(float));
    net->conv4.input = net->relu3.output;
    conv_op_forward(&(net->conv4));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv4)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn4.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->bn4.units);
    net->bn4.input = net->conv4.output;
    batch_norm_op_forward(&(net->bn4));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn4)) duration: %.2fs \n", duration);
#endif

    for(int p=0; p< net->bn4.units * BATCH_SIZE; p++)
    {
        if(net->bn4.output[p]<(0-10) | net->bn4.output[p]>10 )
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: BatchNorm4 error !!! %f\n", net->bn4.output[p]);
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu4.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->relu4.units);
    net->relu4.input = net->bn4.output;
    relu_op_forward(&(net->relu4));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->relu4)) duration: %.2fs \n", duration);
#endif

    //printf(">>>>>>>>>>>>>>>>>>conv5>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv5.output = (float *)calloc(BATCH_SIZE * net->conv5.out_units, sizeof(float));
    net->conv5.input = net->relu4.output;
    conv_op_forward(&(net->conv5));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv5)) duration: %.2fs \n", duration);
#endif
    
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn5.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->bn5.units);
    net->bn5.input = net->conv5.output;
    batch_norm_op_forward(&(net->bn5));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn5)) duration: %.2fs \n", duration);
#endif

    for(int p=0; p< net->bn5.units * BATCH_SIZE; p++)
    {
        if(net->bn5.output[p]<(0-10) | net->bn5.output[p]>10 )
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: BatchNorm5 error !!! %f\n", net->bn5.output[p]);
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu5.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->relu5.units);
    net->relu5.input = net->bn5.output;
    relu_op_forward(&(net->relu5));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->relu5)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp5.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->mp5.out_units);
    net->mp5.input = net->relu5.output;
    max_pooling_op_forward(&(net->mp5));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->mp5)) duration: %.2fs \n", duration);
#endif

    dropout(net->mp5.output, DROPOUT_PROB, net->mp5.out_units);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc1.output = (float *)calloc(BATCH_SIZE * net->fc1.out_units, sizeof(float));
    net->fc1.input = net->mp5.output;
    fc_op_forward(&(net->fc1));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc1)) duration: %.2fs \n", duration);
#endif

    for(int p=0; p< net->fc1.out_units * BATCH_SIZE; p++)
    {
        if(net->fc1.output[p]<(0-64) | net->fc1.output[p]>64)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: fc1 error !!! %d <%.4f> \n", p, net->fc1.output[p]);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu6.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->relu6.units);
    net->relu6.input = net->fc1.output;
    relu_op_forward(&(net->relu6));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->relu6)) duration: %.2fs \n", duration);
#endif
    
    dropout(net->relu6.output, DROPOUT_PROB, net->relu6.units);

    for(int p=0; p< net->relu6.units * BATCH_SIZE; p++)
    {
        if(net->relu6.output[p]<0)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: relu6 error !!! \n");
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc2.output = (float *)calloc(BATCH_SIZE * net->fc2.out_units, sizeof(float));
    net->fc2.input = net->fc1.output;
    fc_op_forward(&(net->fc2));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc2)) duration: %.2fs \n", duration);
#endif

    for(int p=0; p< net->fc2.out_units * BATCH_SIZE; p++)
    {
        if(net->fc2.output[p]<(0-64) | net->fc2.output[p]>64)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: fc2 error !!! %f \n", net->fc2.output[p]);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu7.output = (float *)malloc(BATCH_SIZE * sizeof(float) * net->relu7.units);
    net->relu7.input = net->fc2.output;
    relu_op_forward(&(net->relu7));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->relu7)) duration: %.2fs \n", duration);
#endif
    
    for(int p=0; p< net->relu7.units * BATCH_SIZE; p++)
    {
        if(net->relu7.output[p]<0)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: relu7 error !!! \n");
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc3.output = (float *)calloc(BATCH_SIZE * net->fc3.out_units, sizeof(float));
    net->fc3.input = net->fc2.output;
    fc_op_forward(&(net->fc3));
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc3) duration: %.2fs \n", duration);
#endif

    net->output = net->fc3.output;
}


void alexnet_backward(alexnet *net, int *batch_Y)
{
    alexnet_calloc_d_params(net);

    net->fc3.d_output = (float *)calloc(net->fc3.out_units, sizeof(float));
    cross_entropy_loss(net->fc3.d_output, net->output, batch_Y, OUT_LAYER);

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward cross_entropy duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc3.d_input = (float *)calloc(net->fc3.in_units, sizeof(float));
    fc_op_backward(&(net->fc3));
    free(net->fc3.d_output);
    free(net->fc3.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->fc3)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu7.d_input = (float *)calloc(net->relu7.units, sizeof(float));
    net->relu7.d_output = net->fc3.d_input;
    relu_op_backward(&(net->relu7));
    free(net->relu7.d_output);
    free(net->relu7.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->relu7)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc2.d_input = (float *)calloc(net->fc2.in_units, sizeof(float));
    net->fc2.d_output = net->relu7.d_input;
    fc_op_backward(&(net->fc2));
    free(net->fc2.d_output);
    free(net->fc2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->fc2)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu6.d_input = (float *)calloc(net->relu6.units, sizeof(float));
    net->relu6.d_output = net->fc2.d_input;
    relu_op_backward(&(net->relu6));
    free(net->relu6.d_output);
    free(net->relu6.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->relu6)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->fc1.d_input = (float *)calloc(net->fc1.in_units, sizeof(float));
    net->fc1.d_output = net->relu6.d_input;
    fc_op_backward(&(net->fc1));
    free(net->fc1.d_output);
    free(net->fc1.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->fc1)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp5.d_input = (float *)calloc(net->mp5.in_units, sizeof(float));
    net->mp5.d_output = net->fc1.d_input;
    max_pooling_op_backward(&(net->mp5));
    free(net->mp5.d_output);
    free(net->mp5.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->mp5)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu5.d_input = (float *)calloc(net->relu5.units, sizeof(float));
    net->relu5.d_output = net->mp5.d_input;
    relu_op_backward(&(net->relu5));
    free(net->relu5.d_output);
    free(net->relu5.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->relu5)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn5.d_input = (float *)calloc(net->bn5.units, sizeof(float));
    net->bn5.d_output = net->relu5.d_input;
    batch_norm_op_backward(&(net->bn5));
    free(net->bn5.d_output);
    free(net->bn5.output);
    for(int i=0; i< net->bn5.units; i++)
    {
        if(net->bn5.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn5  %f\n",net->bn5.d_input[i]);
            exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn5)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv5.d_input = (float *)calloc(net->conv5.in_units, sizeof(float));
    net->conv5.d_output = net->bn5.d_input;
    conv_op_backward(&(net->conv5));
    free(net->conv5.d_output);
    free(net->conv5.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv5)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu4.d_input = (float *)calloc(net->relu4.units, sizeof(float));
    net->relu4.d_output = net->conv5.d_input;
    relu_op_backward(&(net->relu4));
    for(int i=0; i< net->relu4.units; i++)
    {
        if(net->relu4.d_output[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! delta:relu4  %.2f\n",net->relu4.d_output[i]);
            break;
        }
    }
    free(net->relu4.d_output);
    free(net->relu4.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->relu4)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn4.d_input = (float *)calloc(net->bn4.units, sizeof(float));
    net->bn4.d_output = net->relu4.d_input;
    batch_norm_op_backward(&(net->bn4));
    free(net->bn4.d_output);
    free(net->bn4.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn4)) duration: %.2fs \n", duration);
#endif

    for(int i=0; i< net->bn4.units; i++)
    {
        if(net->bn4.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn4  %f\n",net->bn4.d_input[i]);
            exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv4.d_input = (float *)calloc(net->conv4.in_units, sizeof(float));
    net->conv4.d_output = net->bn4.d_input;
    conv_op_backward(&(net->conv4));
    free(net->conv4.d_output);
    free(net->conv4.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv4)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu3.d_input = (float *)calloc(net->relu3.units, sizeof(float));
    net->relu3.d_output = net->conv4.d_input;
    relu_op_backward(&(net->relu3));
    for(int i=0; i< net->relu3.units; i++)
    {
        if(net->relu3.d_output[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! delta:relu3  %.2f\n",net->relu3.d_output[i]);
            break;
        }
    }
    free(net->relu3.d_output);
    free(net->relu3.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->relu3)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn3.d_input = (float *)calloc(net->bn3.units, sizeof(float));
    net->bn3.d_output = net->relu3.d_input;
    batch_norm_op_backward(&(net->bn3));
    free(net->bn3.d_output);
    free(net->bn3.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn3)) duration: %.2fs \n", duration);
#endif

    for(int i=0; i< net->bn3.units; i++)
    {
        if(net->bn3.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn3  %f\n",net->bn3.d_input[i]);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv3.d_input = (float *)calloc(net->conv3.in_units, sizeof(float));
    net->conv3.d_output = net->bn3.d_input;
    conv_op_backward(&(net->conv3));
    free(net->conv3.d_output);
    free(net->conv3.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv3)) duration: %.2fs \n", duration);
#endif

    for(int i=0; i< net->conv3.in_units; i++)
    {
        if(net->conv3.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! conv3  %f\n",net->conv3.d_input[i]);
            //exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp2.d_input = (float *)calloc(net->mp2.in_units, sizeof(float));
    net->mp2.d_output = net->conv3.d_input;
    max_pooling_op_backward(&(net->mp2));
    free(net->mp2.d_output);
    free(net->mp2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->mp2)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu2.d_input = (float *)calloc(net->relu2.units, sizeof(float));
    net->relu2.d_output = net->mp2.d_input;
    relu_op_backward(&(net->relu2));
    free(net->relu2.d_output);
    free(net->relu2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->relu2)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn2.d_input = (float *)calloc(net->bn2.units, sizeof(float));
    net->bn2.d_output = net->relu2.d_input;
    batch_norm_op_backward(&(net->bn2));
    free(net->bn2.d_output);
    free(net->bn2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn2)) duration: %.2fs \n", duration);
#endif

    for(int i=0; i< net->bn2.units; i++)
    {
        if(net->bn2.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn2  %f\n",net->bn2.d_input[i]);
            //exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv2.d_input = (float *)calloc(net->conv2.in_units, sizeof(float));
    net->conv2.d_output = net->bn2.d_input;
    conv_op_backward(&(net->conv2));
    free(net->conv2.d_output);
    free(net->conv2.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv2)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->mp1.d_input = (float *)calloc(net->mp1.in_units, sizeof(float));
    net->mp1.d_output = net->conv2.d_input;
    max_pooling_op_backward(&(net->mp1));
    free(net->mp1.d_output);
    free(net->mp1.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->mp1)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->relu1.d_input = (float *)calloc(net->relu1.units, sizeof(float));
    net->relu1.d_output = net->mp1.d_input;
    relu_op_backward(&(net->relu1));
    free(net->relu1.d_output);
    free(net->relu1.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->relu1)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->bn1.d_input = (float *)calloc(net->bn1.units, sizeof(float));
    net->bn1.d_output = net->relu1.d_input;
    batch_norm_op_backward(&(net->bn1));
    free(net->bn1.d_output);
    free(net->bn1.output);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->bn1)) duration: %.2fs \n", duration);
#endif

    for(int i=0; i< net->bn1.units; i++)
    {
        if(net->bn1.d_input[i] > 4)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn1  %f\n",net->bn1.d_input[i]);
            //exit(-1);            
            break;
        }
    }

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    net->conv1.d_input = (float *)calloc(net->conv1.in_units, sizeof(float));
    net->conv1.d_output = net->bn1.d_input;
    conv_op_backward(&(net->conv1));
    free(net->conv1.d_output);
    free(net->conv1.output);
    free(net->conv1.d_input);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward (&(net->conv1)) duration: %.2fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    update_params(net);
#ifdef SHOW_OP_TIME
    clock_gettime(CLOCK_MONOTONIC, &finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" backward update_params(net) duration: %.2fs \n", duration);
#endif

    alexnet_free_d_params(net);
}


void alexnet_malloc_params(alexnet *net)
{

    net->conv1.weights = (float *)malloc(sizeof(float) * C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L);
    net->conv2.weights = (float *)malloc(sizeof(float) * C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L);
    net->conv3.weights = (float *)malloc(sizeof(float) * C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L);
    net->conv4.weights = (float *)malloc(sizeof(float) * C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L);
    net->conv5.weights = (float *)malloc(sizeof(float) * C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L);
    net->fc1.weights = (float *)malloc(sizeof(float) * C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L);
    net->fc2.weights = (float *)malloc(sizeof(float) * FC6_LAYER*FC7_LAYER);
    net->fc3.weights = (float *)malloc(sizeof(float) * FC7_LAYER*OUT_LAYER);

    net->conv1.bias = (float *)malloc(sizeof(float) * C1_CHANNELS);
    net->conv2.bias = (float *)malloc(sizeof(float) * C2_CHANNELS);
    net->conv3.bias = (float *)malloc(sizeof(float) * C3_CHANNELS);
    net->conv4.bias = (float *)malloc(sizeof(float) * C4_CHANNELS);
    net->conv5.bias = (float *)malloc(sizeof(float) * C5_CHANNELS);
    net->fc1.bias = (float *)malloc(sizeof(float) * FC6_LAYER);
    net->fc2.bias = (float *)malloc(sizeof(float) * FC7_LAYER);
    net->fc3.bias = (float *)malloc(sizeof(float) * OUT_LAYER);

    net->bn1.gamma = (float *)malloc(sizeof(float) * C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    net->bn2.gamma = (float *)malloc(sizeof(float) * C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    net->bn3.gamma = (float *)malloc(sizeof(float) * C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    net->bn4.gamma = (float *)malloc(sizeof(float) * C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    net->bn5.gamma = (float *)malloc(sizeof(float) * C5_CHANNELS*FEATURE5_L*FEATURE5_L);

    net->bn1.beta = (float *)malloc(sizeof(float) * C1_CHANNELS*FEATURE1_L*FEATURE1_L);
    net->bn2.beta = (float *)malloc(sizeof(float) * C2_CHANNELS*FEATURE2_L*FEATURE2_L);
    net->bn3.beta = (float *)malloc(sizeof(float) * C3_CHANNELS*FEATURE3_L*FEATURE3_L);
    net->bn4.beta = (float *)malloc(sizeof(float) * C4_CHANNELS*FEATURE4_L*FEATURE4_L);
    net->bn5.beta = (float *)malloc(sizeof(float) * C5_CHANNELS*FEATURE5_L*FEATURE5_L);
}

void alexnet_calloc_d_params(alexnet *net)
{

    net->conv1.d_weights = (float *)calloc( C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L, sizeof(float));
    net->conv2.d_weights = (float *)calloc( C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L, sizeof(float));
    net->conv3.d_weights = (float *)calloc( C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L, sizeof(float));
    net->conv4.d_weights = (float *)calloc( C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L, sizeof(float));
    net->conv5.d_weights = (float *)calloc( C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L, sizeof(float));
    net->fc1.d_weights = (float *)calloc( C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L, sizeof(float));
    net->fc2.d_weights = (float *)calloc( FC6_LAYER*FC7_LAYER, sizeof(float));
    net->fc3.d_weights = (float *)calloc( FC7_LAYER*OUT_LAYER, sizeof(float));

    net->conv1.d_bias = (float *)calloc( C1_CHANNELS, sizeof(float));
    net->conv2.d_bias = (float *)calloc( C2_CHANNELS, sizeof(float));
    net->conv3.d_bias = (float *)calloc( C3_CHANNELS, sizeof(float));
    net->conv4.d_bias = (float *)calloc( C4_CHANNELS, sizeof(float));
    net->conv5.d_bias = (float *)calloc( C5_CHANNELS, sizeof(float));
    net->fc1.d_bias = (float *)calloc( FC6_LAYER, sizeof(float));
    net->fc2.d_bias = (float *)calloc( FC7_LAYER, sizeof(float));
    net->fc3.d_bias = (float *)calloc( OUT_LAYER, sizeof(float));

    net->bn1.d_gamma = (float *)calloc( C1_CHANNELS*FEATURE1_L*FEATURE1_L, sizeof(float));
    net->bn2.d_gamma = (float *)calloc( C2_CHANNELS*FEATURE2_L*FEATURE2_L, sizeof(float));
    net->bn3.d_gamma = (float *)calloc( C3_CHANNELS*FEATURE3_L*FEATURE3_L, sizeof(float));
    net->bn4.d_gamma = (float *)calloc( C4_CHANNELS*FEATURE4_L*FEATURE4_L, sizeof(float));
    net->bn5.d_gamma = (float *)calloc( C5_CHANNELS*FEATURE5_L*FEATURE5_L, sizeof(float));

    net->bn1.d_beta = (float *)calloc( C1_CHANNELS*FEATURE1_L*FEATURE1_L, sizeof(float));
    net->bn2.d_beta = (float *)calloc( C2_CHANNELS*FEATURE2_L*FEATURE2_L, sizeof(float));
    net->bn3.d_beta = (float *)calloc( C3_CHANNELS*FEATURE3_L*FEATURE3_L, sizeof(float));
    net->bn4.d_beta = (float *)calloc( C4_CHANNELS*FEATURE4_L*FEATURE4_L, sizeof(float));
    net->bn5.d_beta = (float *)calloc( C5_CHANNELS*FEATURE5_L*FEATURE5_L, sizeof(float));

}

void alexnet_free_params(alexnet *net)
{
    free(net->conv1.weights);
    free(net->conv2.weights);
    free(net->conv3.weights);
    free(net->conv4.weights);
    free(net->conv5.weights);
    free(net->fc1.weights); 
    free(net->fc2.weights);
    free(net->fc3.weights);

    free(net->conv1.bias); 
    free(net->conv2.bias);
    free(net->conv3.bias); 
    free(net->conv4.bias); 
    free(net->conv5.bias);
    free(net->fc1.bias); 
    free(net->fc2.bias); 
    free(net->fc3.bias);

    free(net->bn1.gamma); 
    free(net->bn2.gamma); 
    free(net->bn3.gamma); 
    free(net->bn4.gamma);
    free(net->bn5.gamma);

    free(net->bn1.beta);
    free(net->bn2.beta);
    free(net->bn3.beta);
    free(net->bn4.beta);
    free(net->bn5.beta); 
}

void alexnet_free_d_params(alexnet *net)
{
    free(net->conv1.d_weights);
    free(net->conv2.d_weights);
    free(net->conv3.d_weights);
    free(net->conv4.d_weights);
    free(net->conv5.d_weights);
    free(net->fc1.d_weights); 
    free(net->fc2.d_weights);
    free(net->fc3.d_weights);

    free(net->conv1.d_bias); 
    free(net->conv2.d_bias);
    free(net->conv3.d_bias); 
    free(net->conv4.d_bias); 
    free(net->conv5.d_bias);
    free(net->fc1.d_bias); 
    free(net->fc2.d_bias); 
    free(net->fc3.d_bias);

    free(net->bn1.d_gamma); 
    free(net->bn2.d_gamma); 
    free(net->bn3.d_gamma); 
    free(net->bn4.d_gamma);
    free(net->bn5.d_gamma);

    free(net->bn1.d_beta);
    free(net->bn2.d_beta);
    free(net->bn3.d_beta);
    free(net->bn4.d_beta);
    free(net->bn5.d_beta); 
}

void alexnet_param_init(alexnet *net)
{
    net->conv1.in_channels = IN_CHANNELS;
    net->conv1.out_channels = C1_CHANNELS;
    net->conv1.in_h = FEATURE0_L;
    net->conv1.in_w = FEATURE0_L;
    net->conv1.kernel_size = C1_KERNEL_L;
    net->conv1.padding = C1_PADDING;
    net->conv1.stride = C1_STRIDES;
    net->conv1.out_h = FEATURE1_L;
    net->conv1.out_w = FEATURE1_L;
    net->conv1.in_units = IN_CHANNELS*FEATURE0_L*FEATURE0_L;
    net->conv1.out_units = C1_CHANNELS*FEATURE1_L*FEATURE1_L;

    net->bn1.units = net->conv1.out_units;

    net->relu1.units = net->bn1.units;
    
    net->mp1.channels = C1_CHANNELS;
    net->mp1.stride = 2;
    net->mp1.kernel_size = 3;
    net->mp1.in_h = FEATURE1_L;
    net->mp1.in_w = FEATURE1_L;
    net->mp1.out_w = POOLING1_L;
    net->mp1.out_h = POOLING1_L;
    net->mp1.in_units = net->relu1.units;
    net->mp1.out_units = C1_CHANNELS*POOLING1_L*POOLING1_L;

    net->conv2.in_channels = C1_CHANNELS;
    net->conv2.out_channels = C2_CHANNELS;
    net->conv2.in_h = POOLING1_L;
    net->conv2.in_w = POOLING1_L;
    net->conv2.kernel_size = C2_KERNEL_L;
    net->conv2.padding = C2_PADDING;
    net->conv2.stride = C2_STRIDES;
    net->conv2.out_h = FEATURE2_L;
    net->conv2.out_w = FEATURE2_L;
    net->conv2.in_units = net->mp1.out_units;
    net->conv2.out_units = C2_CHANNELS*FEATURE2_L*FEATURE2_L;

    net->bn2.units = net->conv2.out_units;

    net->relu2.units = net->bn2.units;

    net->mp2.channels = C2_CHANNELS;
    net->mp2.stride = 2;
    net->mp2.kernel_size = 3;
    net->mp2.in_h = FEATURE2_L;
    net->mp2.in_w = FEATURE2_L;
    net->mp2.out_w = POOLING2_L;
    net->mp2.out_h = POOLING2_L;
    net->mp2.in_units = net->relu2.units;
    net->mp2.out_units = C2_CHANNELS*POOLING2_L*POOLING2_L;

    net->conv3.in_channels = C2_CHANNELS;
    net->conv3.out_channels = C3_CHANNELS;
    net->conv3.in_h = POOLING2_L;
    net->conv3.in_w = POOLING2_L;
    net->conv3.kernel_size = C3_KERNEL_L;
    net->conv3.padding = C3_PADDING;
    net->conv3.stride = C3_STRIDES;
    net->conv3.out_h = FEATURE3_L;
    net->conv3.out_w = FEATURE3_L;
    net->conv3.in_units = net->mp2.out_units;
    net->conv3.out_units = C3_CHANNELS*FEATURE3_L*FEATURE3_L;

    net->bn3.units = net->conv3.out_units;

    net->relu3.units = net->bn3.units;

    net->conv4.in_channels = C3_CHANNELS;
    net->conv4.out_channels = C4_CHANNELS;
    net->conv4.in_h = FEATURE3_L;
    net->conv4.in_w = FEATURE3_L;
    net->conv4.kernel_size = C4_KERNEL_L;
    net->conv4.padding = C4_PADDING;
    net->conv4.stride = C4_STRIDES;
    net->conv4.out_h = FEATURE4_L;
    net->conv4.out_w = FEATURE4_L;
    net->conv4.in_units = net->relu3.units;
    net->conv4.out_units = C4_CHANNELS*FEATURE4_L*FEATURE4_L;

    net->bn4.units = net->conv4.out_units;

    net->relu4.units = net->bn4.units;

    net->conv5.in_channels = C4_CHANNELS;
    net->conv5.out_channels = C5_CHANNELS;
    net->conv5.in_h = FEATURE5_L;
    net->conv5.in_w = FEATURE5_L;
    net->conv5.kernel_size = C5_KERNEL_L;
    net->conv5.padding = C5_PADDING;
    net->conv5.stride = C5_STRIDES;
    net->conv5.out_h = FEATURE5_L;
    net->conv5.out_w = FEATURE5_L;
    net->conv5.in_units = net->relu4.units;
    net->conv5.out_units = C5_CHANNELS*FEATURE5_L*FEATURE5_L;

    net->bn5.units = net->conv5.out_units;

    net->relu5.units = net->bn5.units;

    net->mp5.channels = C5_CHANNELS;
    net->mp5.stride = 2;
    net->mp5.kernel_size = 3;
    net->mp5.in_h = FEATURE5_L;
    net->mp5.in_w = FEATURE5_L;
    net->mp5.out_w = POOLING5_L;
    net->mp5.out_h = POOLING5_L;
    net->mp5.in_units = net->relu5.units;
    net->mp5.out_units = C5_CHANNELS*POOLING5_L*POOLING5_L;

    net->fc1.in_units = net->mp5.out_units;
    net->fc1.out_units = FC6_LAYER;
    
    net->relu6.units = FC6_LAYER; 
    
    net->fc2.in_units = FC6_LAYER;
    net->fc2.out_units = FC7_LAYER;

    net->relu7.units = FC7_LAYER;

    net->fc3.in_units = FC7_LAYER;
    net->fc3.out_units = OUT_LAYER;


    gauss_initialization(net->conv1.weights, C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L, net->conv1.in_units, net->conv1.out_units);
    gauss_initialization(net->conv2.weights, C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L, net->conv2.in_units, net->conv2.out_units);
    gauss_initialization(net->conv3.weights, C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L, net->conv3.in_units, net->conv3.out_units);
    gauss_initialization(net->conv4.weights, C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L, net->conv4.in_units, net->conv4.out_units);
    gauss_initialization(net->conv5.weights, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L, net->conv5.in_units, net->conv5.out_units);
    gauss_initialization(net->fc1.weights, C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L, net->fc1.in_units, net->fc1.out_units);
    gauss_initialization(net->fc2.weights, FC6_LAYER*FC7_LAYER, net->fc2.in_units, net->fc2.out_units);
    gauss_initialization(net->fc3.weights, FC7_LAYER*OUT_LAYER, net->fc3.in_units, net->fc3.out_units);

    int i;
    for(i=0; i<C1_CHANNELS; i++)
        net->conv1.bias[i] = 1;
    for(i=0; i<C2_CHANNELS; i++)
        net->conv2.bias[i] = 1;
    for(i=0; i<C3_CHANNELS; i++)
        net->conv3.bias[i] = 1;
    for(i=0; i<C4_CHANNELS; i++)
        net->conv4.bias[i] = 1;
    for(i=0; i<C5_CHANNELS; i++)
        net->conv5.bias[i] = 1;
    for(i=0; i<FC6_LAYER; i++)
        net->fc1.bias[i] = 1;
    for(i=0; i<FC7_LAYER; i++)
        net->fc2.bias[i] = 1;
    for(i=0; i<OUT_LAYER; i++)
        net->fc3.bias[i] = 1;

    for(i=0; i<(net->bn1.units); i++)
    {
        net->bn1.gamma[i] = 1;
        net->bn1.beta[i] = 0;
    }
    for(i=0; i<(net->bn2.units); i++)
    {
        net->bn2.gamma[i] = 1;
        net->bn2.beta[i] = 0;
    }
    for(i=0; i<(net->bn3.units); i++)
    {
        net->bn3.gamma[i] = 1;
        net->bn3.beta[i] = 0;
    }

    for(i=0; i<(net->bn4.units); i++)
    {
        net->bn4.gamma[i] = 1;
        net->bn4.beta[i] = 0;
    }

    for(i=0; i<(net->bn5.units); i++)
    {
        net->bn5.gamma[i] = 1;
        net->bn5.beta[i] = 0;
    }

}


static float U_Random()
{
    float f;
    srand( (unsigned)time( NULL ) );
    f = (float)(rand() % 100);
    return f/100;
}

void gauss_initialization(float *p, int n, int in_units, int out_units)
{
    float mean  = 0;
    float stddv = 0.01;

	float V1, V2, S;
	static int phase = 0;
	float X;
    for(int shift=0; shift<n; shift++)
    {
        if(phase == 0) 
        {
            do {
                float U1 = (float) rand() / RAND_MAX;
                float U2 = (float) rand() / RAND_MAX;

                V1 = 2 * U1 - 1;
                V2 = 2 * U2 - 1;
                S = V1 * V1 + V2 * V2;
            } while (S >= 1 || S == 0);
    
            X = V1 * sqrt(-2 * log(S) / S);
        }else
        {
            X = V2 * sqrt(-2 * log(S) / S);
        }
        phase = 1 - phase;

        p[shift] = mean + stddv * X;
    }
}
