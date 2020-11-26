#include <stdlib.h>
#include "alexnet.h"


void predict(Alexnet *alexnet, float *inputs, float *outputs);
void train(Alexnet *alexnet, int epochs);
