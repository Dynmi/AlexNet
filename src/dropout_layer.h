//
// File:        dropout_layer.h
// Description: interface of dropout layer
// Author:      Haris Wang
//
// Inverted dropout with a stored mask for the backward pass: during
// training each element is kept with probability keep=1-prob and rescaled
// by 1/keep so the expected activation is unchanged and eval needs no
// rescaling (eval simply skips dropout_forward entirely -- see
// forward_alexnet's is_training checks).
//
// dropout_forward fills `mask` (1/keep for kept units, 0 for dropped) and
// scales x in place. dropout_backward multiplies the incoming gradient by
// the same mask, so it must be called with the mask produced by the
// matching forward pass on the same batch.
#include <stdlib.h>

void dropout_forward(float *x, float *mask, float prob, int n);
void dropout_backward(float *dx, const float *mask, int n);
