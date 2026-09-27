/*
 * single_conv.c
 *
 * Train a single Conv layer (Conv2d 3→10, Global Average Pool, Cross-Entropy)
 * on CIFAR-10 — mirrors the Python conv1layer.py script.
 *
 * Architecture:
 *   Input [3, 32, 32]
 *     → Conv(in=3, out=10, kernel=3, stride=1, no padding)  → [10, 30, 30]
 *     → Global Average Pool                                  → [10]
 *     → Softmax + Cross-Entropy Loss
 *
 * Compile from project root:
 *   gcc -O2 -o single_conv src/single_conv.c src/data.c \
 *       src/convolution_layer.c src/matrix.c              \
 *       -I src -lm -lpthread
 *
 * Run from project root:
 *   ./single_conv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "convolution_layer.h"
#include "data.h"

/* ------------------------------------------------------------------ */
/* Hyper-parameters                                                     */
/* ------------------------------------------------------------------ */
#define IN_C        3
#define NUM_CLASSES 10       /* = out_channels = number of kernels */
#define KERNEL      3
#define STRIDE      1

#define IMG_W       32
#define IMG_H       32

/* No padding → output shrinks by (kernel-1) on each side */
#define OUT_W       (IMG_W - KERNEL + 1)   /* 30 */
#define OUT_H       (IMG_H - KERNEL + 1)   /* 30 */

#define BATCH_SIZE  64
#define NUM_EPOCHS  20
#define LR          0.01f    /* matches LEARNING_RATE in train.c */
#define MOMENTUM    0.9f

/* ------------------------------------------------------------------ */
/* Weight initialisation — matches gauss_initialization() in alexnet.c */
/* Box-Muller, mean=0, std=0.01 (fixed, same as rest of codebase)      */
/* ------------------------------------------------------------------ */
static void gauss_init(float *p, int n)
{
    float stddv = 0.01f;
    static int phase = 0;
    float V1, V2, S, X;
    for (int i = 0; i < n; i++) {
        if (phase == 0) {
            do {
                float U1 = (float)rand() / RAND_MAX;
                float U2 = (float)rand() / RAND_MAX;
                V1 = 2*U1 - 1;  V2 = 2*U2 - 1;
                S  = V1*V1 + V2*V2;
            } while (S >= 1 || S == 0);
            X = V1 * sqrtf(-2.0f * logf(S) / S);
        } else {
            X = V2 * sqrtf(-2.0f * logf(S) / S);
        }
        phase = 1 - phase;
        p[i]  = stddv * X;
    }
}

/* ------------------------------------------------------------------ */
/* Global Average Pool  forward                                         */
/*   in  : [batch, C, H, W]  (C*H*W contiguous per image)             */
/*   out : [batch, C]                                                   */
/* ------------------------------------------------------------------ */
static void gap_forward(const float *in, float *out,
                        int batch, int C, int H, int W)
{
    int hw = H * W;
    for (int b = 0; b < batch; b++) {
        const float *inp = in  + b * C * hw;
        float       *op  = out + b * C;
        for (int c = 0; c < C; c++) {
            float s = 0;
            const float *ch = inp + c * hw;
            for (int i = 0; i < hw; i++) s += ch[i];
            op[c] = s / (float)hw;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Global Average Pool  backward                                        */
/*                                                                      */
/* NOTE: conv_op_backward expects d_output to be [out_units] for a     */
/* SINGLE (averaged) gradient, not [batch * out_units].  The backward  */
/* routine accumulates weight gradients over all images using this      */
/* single averaged gradient, matching the pattern used throughout       */
/* the rest of the codebase.                                            */
/*                                                                      */
/*   d_gap_avg : [C]         averaged gradient from cross-entropy       */
/*   d_in_avg  : [C, H, W]   gradient to feed to conv backward         */
/*                                                                      */
/* Gradient of mean: d_in[c, p] = d_out[c] / (H*W)                    */
/* ------------------------------------------------------------------ */
static void gap_backward_avg(const float *d_gap_avg, float *d_in_avg,
                             int C, int H, int W)
{
    int hw = H * W;
    for (int c = 0; c < C; c++) {
        float grad = d_gap_avg[c] / (float)hw;
        float *ch  = d_in_avg + c * hw;
        for (int i = 0; i < hw; i++) ch[i] = grad;
    }
}

/* ------------------------------------------------------------------ */
/* Softmax + Cross-Entropy loss                                         */
/*                                                                      */
/*   logits     : [batch, classes]  (raw conv output after GAP)        */
/*   d_logits   : [classes]         OUTPUT: averaged gradient (zeroed  */
/*                before this call)                                     */
/*   returns      mean CE loss over the batch                          */
/* ------------------------------------------------------------------ */
static float cross_entropy(const float *logits, float *d_logits,
                            const int *labels, int classes, int batch)
{
    float ce_loss = 0;
    for (int p = 0; p < batch; p++) {
        const float *row = logits + p * classes;

        /* numerical stability: subtract max before exp */
        float mx = row[0];
        for (int i = 1; i < classes; i++) if (row[i] > mx) mx = row[i];

        float esum = 0;
        for (int i = 0; i < classes; i++) esum += expf(row[i] - mx);

        ce_loss += -logf(expf(row[labels[p]] - mx) / esum);

        /* softmax gradient (averaged over batch at the end) */
        for (int i = 0; i < classes; i++) {
            float sm = expf(row[i] - mx) / esum;
            d_logits[i] += (i == labels[p]) ? sm - 1.0f : sm;
        }
    }
    for (int i = 0; i < classes; i++) d_logits[i] /= (float)batch;
    return ce_loss / (float)batch;
}

/* ------------------------------------------------------------------ */
/* Momentum SGD  — identical to train.c: velocity clipped to [-1, 1]  */
/* ------------------------------------------------------------------ */
#define CLIP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))

static void momentum_sgd(float *w, float *v, const float *dw, int n)
{
    for (int i = 0; i < n; i++) {
        v[i] = MOMENTUM * v[i] - LR * dw[i];
        v[i] = CLIP(v[i], -1.0f, 1.0f);   /* matches CLIP() in train.c */
        w[i] += v[i];
    }
}

/* ------------------------------------------------------------------ */
/* Evaluate accuracy on a list file (reads sequentially)               */
/* ------------------------------------------------------------------ */
static float evaluate(FILE *fp_val, conv_op *conv,
                      float *X_buf, float *gap_buf)
{
    int correct = 0, total = 0;
    int labels[BATCH_SIZE];
    int owoh = OUT_W * OUT_H;

    rewind(fp_val);
    while (1) {
        int n = 0;
        for (; n < BATCH_SIZE; n++) {
            int lbl; char path[512];
            if (fscanf(fp_val, "%d %s", &lbl, path) != 2) goto done_eval;
            image img = load_image(path, IMG_W, IMG_H, IN_C, /*flip=*/0);
            memcpy(X_buf + n * IN_C * IMG_W * IMG_H,
                   img.data, IN_C * IMG_W * IMG_H * sizeof(float));
            free_image(&img);
            labels[n] = lbl;
        }

        conv->batchsize = n;
        conv->input     = X_buf;
        conv->output    = (float *)calloc((size_t)n * NUM_CLASSES * owoh, sizeof(float));
        conv_op_forward(conv);

        /* conv_op_forward allocates input_col; free it (no backward here) */
        free(conv->input_col);
        conv->input_col = NULL;

        gap_forward(conv->output, gap_buf, n, NUM_CLASSES, OUT_H, OUT_W);
        free(conv->output);
        conv->output = NULL;

        for (int b = 0; b < n; b++) {
            float *row = gap_buf + b * NUM_CLASSES;
            int best = 0;
            for (int c = 1; c < NUM_CLASSES; c++)
                if (row[c] > row[best]) best = c;
            if (best == labels[b]) correct++;
            total++;
        }
    }
done_eval:
    return total > 0 ? 100.0f * (float)correct / (float)total : 0.0f;
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
int main(void)
{
    srand((unsigned)time(NULL));

    /* ---- Layer setup ---- */
    conv_op conv;
    memset(&conv, 0, sizeof(conv_op));
    conv.in_channels  = IN_C;
    conv.out_channels = NUM_CLASSES;
    conv.kernel_size  = KERNEL;
    conv.stride       = STRIDE;
    conv.padding      = 0;
    conv.in_w         = IMG_W;
    conv.in_h         = IMG_H;
    conv.out_w        = OUT_W;
    conv.out_h        = OUT_H;
    conv.in_units     = IN_C  * IMG_W * IMG_H;    /* 3072  */
    conv.out_units    = NUM_CLASSES * OUT_W * OUT_H; /* 9000 */
    conv.batchsize    = BATCH_SIZE;

    /* ---- Weights ---- */
    calloc_conv_weights(&conv);
    calloc_conv_dweights(&conv);

    int n_w = NUM_CLASSES * IN_C * KERNEL * KERNEL;   /* 10 * 3 * 3 * 3 = 270 */
    gauss_init(conv.weights, n_w);
    /* bias stays zero */

    /* Momentum velocity buffers */
    float *v_w = (float *)calloc(n_w,          sizeof(float));
    float *v_b = (float *)calloc(NUM_CLASSES,   sizeof(float));

    /* ---- I/O buffers ---- */
    int owoh = OUT_W * OUT_H;  /* 900 */

    float *X       = (float *)malloc((size_t)BATCH_SIZE * IN_C * IMG_W * IMG_H * sizeof(float));
    float *gap_out = (float *)malloc((size_t)BATCH_SIZE * NUM_CLASSES * sizeof(float));

    /*
     * d_gap and d_conv_out hold AVERAGED gradients (single-image equivalent),
     * consistent with how conv_op_backward is designed (see comments above).
     */
    float *d_gap      = (float *)malloc(NUM_CLASSES  * sizeof(float));
    float *d_conv_out = (float *)malloc(NUM_CLASSES  * owoh * sizeof(float));

    /* conv backward needs a d_input buffer even though we don't use it
     * (conv is the first layer; col2img writes to it internally)        */
    float *dummy_d_in = (float *)calloc(conv.in_units, sizeof(float));

    int *labels = (int *)malloc(BATCH_SIZE * sizeof(int));

    /* ---- Data files ---- */
    const char *train_list = "./cifar10/images.list";
    const char *val_list   = "./cifar10/image_val.list";

    FILE *fp_train = fopen(train_list, "r");
    FILE *fp_val   = fopen(val_list,   "r");

    if (!fp_train) {
        fprintf(stderr, "Cannot open %s\n", train_list);
        return 1;
    }

    /* Count training images */
    int total_images = 0;
    { char line[512]; while (fgets(line, sizeof(line), fp_train)) total_images++; }
    rewind(fp_train);
    int batches_per_epoch = total_images / BATCH_SIZE;

    /* ---- Banner ---- */
    printf("=== Single Conv Layer on CIFAR-10 ===\n");
    printf("Arch  : Conv(%d→%d, k=%d, s=%d, no-pad) → GAP → Softmax+CE\n",
           IN_C, NUM_CLASSES, KERNEL, STRIDE);
    printf("Output: [%d, %d, %d] → [%d, %d, %d] → [%d]\n",
           IN_C, IMG_H, IMG_W, NUM_CLASSES, OUT_H, OUT_W, NUM_CLASSES);
    printf("Params: batch=%d  epochs=%d  lr=%.4f  momentum=%.2f\n",
           BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM);
    printf("Train : %d images  →  %d batches/epoch\n\n",
           total_images, batches_per_epoch);

    /* ---- Training loop ---- */
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

        rewind(fp_train);
        float epoch_loss    = 0;
        int   epoch_correct = 0, epoch_total = 0;

        for (int b = 0; b < batches_per_epoch; b++) {

            /* 1. Load next batch from disk */
            get_next_batch(BATCH_SIZE, X, labels,
                           IMG_W, IMG_H, IN_C, NUM_CLASSES, fp_train);

            /* 2. Forward: Conv → [batch, NUM_CLASSES, OUT_H, OUT_W] */
            conv.batchsize = BATCH_SIZE;
            conv.input     = X;
            conv.output    = (float *)calloc((size_t)BATCH_SIZE * NUM_CLASSES * owoh,
                                             sizeof(float));
            conv_op_forward(&conv);
            /* input_col is kept alive for the backward pass */

            /* 3. Forward: GAP → [batch, NUM_CLASSES] */
            gap_forward(conv.output, gap_out, BATCH_SIZE, NUM_CLASSES, OUT_H, OUT_W);

            /* 4. Loss + averaged gradient at GAP output */
            memset(d_gap, 0, NUM_CLASSES * sizeof(float));
            float loss = cross_entropy(gap_out, d_gap, labels, NUM_CLASSES, BATCH_SIZE);
            epoch_loss += loss;

            /* Training accuracy */
            for (int p = 0; p < BATCH_SIZE; p++) {
                float *row = gap_out + p * NUM_CLASSES;
                int best = 0;
                for (int c = 1; c < NUM_CLASSES; c++)
                    if (row[c] > row[best]) best = c;
                if (best == labels[p]) epoch_correct++;
                epoch_total++;
            }

            /* 5. Backward: GAP → distribute avg gradient to every spatial pos
             *    d_conv_out[c, p] = d_gap[c] / owoh  */
            gap_backward_avg(d_gap, d_conv_out, NUM_CLASSES, OUT_H, OUT_W);

            /* 6. Backward: Conv (accumulates d_weights & d_bias, frees input_col)
             *
             *    conv_op_backward modifies d_output in-place (transposes it).
             *    We pass a separate copy so d_conv_out stays intact for debug.
             *    d_input is a dummy buffer we discard.                          */
            float *d_conv_out_copy = (float *)malloc(NUM_CLASSES * owoh * sizeof(float));
            memcpy(d_conv_out_copy, d_conv_out, NUM_CLASSES * owoh * sizeof(float));

            memset(conv.d_weights, 0, (size_t)n_w       * sizeof(float));
            memset(conv.d_bias,    0, NUM_CLASSES        * sizeof(float));
            conv.d_output = d_conv_out_copy;
            conv.d_input  = dummy_d_in;

            conv_op_backward(&conv);     /* frees input_col, modifies d_output  */
            free(d_conv_out_copy);

            /* 7. Momentum SGD weight update */
            momentum_sgd(conv.weights, v_w, conv.d_weights, n_w);
            momentum_sgd(conv.bias,    v_b, conv.d_bias,    NUM_CLASSES);

            /* Free this batch's conv output */
            free(conv.output);
            conv.output = NULL;

            if ((b + 1) % 100 == 0)
                printf("  Epoch %2d  batch %4d/%4d  loss=%.4f\n",
                       epoch + 1, b + 1, batches_per_epoch,
                       epoch_loss / (float)(b + 1));
        }

        float train_acc = 100.0f * (float)epoch_correct / (float)epoch_total;
        float val_acc   = fp_val ? evaluate(fp_val, &conv, X, gap_out) : -1.0f;

        printf("Epoch %2d/%2d  avg_loss=%.4f  train_acc=%.2f%%",
               epoch + 1, NUM_EPOCHS,
               epoch_loss / (float)batches_per_epoch,
               train_acc);
        if (val_acc >= 0) printf("  val_acc=%.2f%%", val_acc);
        printf("\n\n");
    }

    /* ---- Cleanup ---- */
    fclose(fp_train);
    if (fp_val) fclose(fp_val);

    free_conv_weights(&conv);
    free_conv_dweights(&conv);
    free(v_w);    free(v_b);
    free(X);      free(gap_out);
    free(d_gap);  free(d_conv_out);
    free(dummy_d_in);
    free(labels);

    return 0;
}
