//
// File:        matrix.h
// Description: interface of matrix computation
// Author:      Haris Wang
//
#include <stdlib.h>

void matrix_multiply(const float *a, const float *b, float *c, const int M, const int N, const int K);
void matrix_transpose(float *x, int m, int n);
