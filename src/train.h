#include <stdlib.h>
#include "alexnet.h"


// Definiation of metric type
#define METRIC_ACCURACY  0
#define METRIC_PRECISION 1      // macro-precision
#define METRIC_RECALL    2      // macro-recall
#define METRIC_F1SCORE   3
#define METRIC_ROC       4


void metrics(float *ret, int *preds, int *labels, 
                int classes, int TotalNum, int type);

void predict(Alexnet *alexnet, float *inputs, float *outputs);

void train(Alexnet *alexnet, int epochs);
