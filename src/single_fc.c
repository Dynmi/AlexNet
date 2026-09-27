/*
 * single_fc.c
 *
 * Train a single Fully-Connected layer on CIFAR-10.
 * Mirrors fc1layer.py exactly in architecture and hyper-parameters.
 *
 * Architecture:
 *   Input  [3072]  (flattened 3x32x32, normalised to [-1, 1])
 *     → FC(3072 → 10)
 *     → Softmax + Cross-Entropy Loss
 *
 * Compile from project root:
 *   gcc -O2 -o single_fc src/single_fc.c src/data.c \
 *       src/fc_layer.c src/matrix.c                  \
 *       -I src -lm -lpthread
 *
 * Run from project root:
 *   ./single_fc
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "fc_layer.h"
#include "data.h"

/* ------------------------------------------------------------------ */
/* Hyper-parameters  (match train.c and fc1layer.py)                   */
/* ------------------------------------------------------------------ */
#define IN_C        3
#define IMG_W       32
#define IMG_H       32
#define IN_UNITS    (IN_C * IMG_W * IMG_H)   /* 3072 */
#define NUM_CLASSES 10

#define BATCH_SIZE  64
#define NUM_EPOCHS  20

#define LR          0.001f   /* 0.01 (train.c) is calibrated for AlexNet+BN; 0.001 suits bare FC */
#define MOMENTUM    0.9f

/* ------------------------------------------------------------------ */
/* Weight init — matches gauss_initialization() in alexnet.c          */
/* Box-Muller, mean=0, std=0.01                                        */
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
/* Softmax + Cross-Entropy loss                                         */
/*                                                                      */
/* Matches cross_entropy_loss() in train.c:                            */
/*   d_logits[i] = (softmax[i] - 1_{i==label}) / batch  averaged      */
/*   returns mean CE loss                                               */
/*                                                                      */
/* NOTE: fc_op_backward expects d_output[out_units] as a SINGLE        */
/* averaged gradient vector (not per-batch).  The routine accumulates  */
/* d_weights internally over all batchsize images.                     */
/* ------------------------------------------------------------------ */
static float cross_entropy(const float *logits, float *d_logits,
                            const int *labels, int classes, int batch)
{
    float ce_loss = 0;
    for (int p = 0; p < batch; p++) {
        const float *row = logits + p * classes;

        float mx = row[0];
        for (int i = 1; i < classes; i++) if (row[i] > mx) mx = row[i];

        float esum = 0;
        for (int i = 0; i < classes; i++) esum += expf(row[i] - mx);

        ce_loss += -logf(expf(row[labels[p]] - mx) / esum);

        for (int i = 0; i < classes; i++) {
            float sm = expf(row[i] - mx) / esum;
            d_logits[i] += (i == labels[p]) ? sm - 1.0f : sm;
        }
    }
    for (int i = 0; i < classes; i++) d_logits[i] /= (float)batch;
    return ce_loss / (float)batch;
}

/* ------------------------------------------------------------------ */
/* Momentum SGD — identical to momentum_sgd + CLIP in train.c         */
/* ------------------------------------------------------------------ */
#define CLIP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))

static void momentum_sgd(float *w, float *v, const float *dw, int n)
{
    for (int i = 0; i < n; i++) {
        v[i] = MOMENTUM * v[i] - LR * dw[i];
        v[i] = CLIP(v[i], -1.0f, 1.0f);
        w[i] += v[i];
    }
}

/* ------------------------------------------------------------------ */
/* Evaluate accuracy on a list file                                    */
/* ------------------------------------------------------------------ */
static float evaluate(FILE *fp_val, fc_op *fc, float *X_buf, float *out_buf)
{
    int correct = 0, total = 0;
    int labels[BATCH_SIZE];

    rewind(fp_val);
    while (1) {
        int n = 0;
        for (; n < BATCH_SIZE; n++) {
            int lbl; char path[512];
            if (fscanf(fp_val, "%d %s", &lbl, path) != 2) goto done_eval;
            image img = load_image(path, IMG_W, IMG_H, IN_C, /*flip=*/0);
            memcpy(X_buf + n * IN_UNITS, img.data, IN_UNITS * sizeof(float));
            free_image(&img);
            labels[n] = lbl;
        }

        fc->batchsize = n;
        fc->input     = X_buf;
        fc->output    = out_buf;
        fc_op_forward(fc);

        for (int b = 0; b < n; b++) {
            float *row = out_buf + b * NUM_CLASSES;
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
    fc_op fc;
    memset(&fc, 0, sizeof(fc_op));
    fc.in_units  = IN_UNITS;
    fc.out_units = NUM_CLASSES;
    fc.batchsize = BATCH_SIZE;

    calloc_fc_weights(&fc);
    calloc_fc_dweights(&fc);

    int n_w = IN_UNITS * NUM_CLASSES;    /* 3072 * 10 = 30720 */
    gauss_init(fc.weights, n_w);
    /* bias stays zero */

    /* Momentum velocity buffers */
    float *v_w = (float *)calloc(n_w,          sizeof(float));
    float *v_b = (float *)calloc(NUM_CLASSES,   sizeof(float));

    /* ---- I/O buffers ---- */
    /* input  : [batch, IN_UNITS]    — flat images, one row per image */
    /* output : [batch, NUM_CLASSES] — logits, one row per image      */
    float *X       = (float *)malloc((size_t)BATCH_SIZE * IN_UNITS    * sizeof(float));
    float *out_buf = (float *)malloc((size_t)BATCH_SIZE * NUM_CLASSES * sizeof(float));
    float *d_out   = (float *)malloc(NUM_CLASSES * sizeof(float));   /* averaged gradient */
    float *d_in    = (float *)calloc(IN_UNITS,    sizeof(float));    /* discarded (first layer) */
    int   *labels  = (int   *)malloc(BATCH_SIZE   * sizeof(int));

    /* ---- Data files ---- */
    const char *train_list = "./cifar10/images.list";
    const char *val_list   = "./cifar10/image_val.list";

    FILE *fp_train = fopen(train_list, "r");
    FILE *fp_val   = fopen(val_list,   "r");
    if (!fp_train) { fprintf(stderr, "Cannot open %s\n", train_list); return 1; }

    /* Count training images */
    int total_images = 0;
    { char line[512]; while (fgets(line, sizeof(line), fp_train)) total_images++; }
    rewind(fp_train);
    int batches_per_epoch = total_images / BATCH_SIZE;

    /* ---- Banner ---- */
    printf("=== Single FC Layer on CIFAR-10 ===\n");
    printf("Arch  : Linear(%d → %d) → Softmax+CE\n", IN_UNITS, NUM_CLASSES);
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

            /* 1. Load batch */
            get_next_batch(BATCH_SIZE, X, labels,
                           IMG_W, IMG_H, IN_C, NUM_CLASSES, fp_train);

            /* 2. Forward */
            fc.batchsize = BATCH_SIZE;
            fc.input     = X;
            fc.output    = out_buf;
            fc_op_forward(&fc);

            /* 3. Loss + averaged gradient at output */
            memset(d_out, 0, NUM_CLASSES * sizeof(float));
            float loss = cross_entropy(out_buf, d_out, labels, NUM_CLASSES, BATCH_SIZE);
            epoch_loss += loss;

            /* Training accuracy */
            for (int p = 0; p < BATCH_SIZE; p++) {
                float *row = out_buf + p * NUM_CLASSES;
                int best = 0;
                for (int c = 1; c < NUM_CLASSES; c++)
                    if (row[c] > row[best]) best = c;
                if (best == labels[p]) epoch_correct++;
                epoch_total++;
            }

            /* 4. Backward
             *
             * fc_op_backward expects:
             *   d_output : [out_units]  — the SINGLE averaged gradient (not per-batch)
             *   d_input  : [in_units]   — written but discarded here (first layer)
             * It accumulates d_weights over all batchsize rows of fc.input internally.
             */
            memset(fc.d_weights, 0, (size_t)n_w        * sizeof(float));
            memset(fc.d_bias,    0,  NUM_CLASSES        * sizeof(float));
            memset(d_in,         0,  IN_UNITS           * sizeof(float));
            fc.d_output = d_out;
            fc.d_input  = d_in;
            fc_op_backward(&fc);

            /* 5. Weight update */
            momentum_sgd(fc.weights, v_w, fc.d_weights, n_w);
            momentum_sgd(fc.bias,    v_b, fc.d_bias,    NUM_CLASSES);

            if ((b + 1) % 100 == 0)
                printf("  Epoch %2d  batch %4d/%4d  loss=%.4f\n",
                       epoch + 1, b + 1, batches_per_epoch,
                       epoch_loss / (float)(b + 1));
        }

        float train_acc = 100.0f * (float)epoch_correct / (float)epoch_total;
        float val_acc   = fp_val ? evaluate(fp_val, &fc, X, out_buf) : -1.0f;

        printf("Epoch %2d/%2d  avg_loss=%.4f  train_acc=%.2f%%",
               epoch + 1, NUM_EPOCHS,
               epoch_loss / (float)batches_per_epoch, train_acc);
        if (val_acc >= 0) printf("  val_acc=%.2f%%", val_acc);
        printf("\n\n");
    }

    /* ---- Cleanup ---- */
    fclose(fp_train);
    if (fp_val) fclose(fp_val);

    free_fc_weights(&fc);
    free_fc_dweights(&fc);
    free(v_w);  free(v_b);
    free(X);    free(out_buf);
    free(d_out); free(d_in);
    free(labels);

    return 0;
}
