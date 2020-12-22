//
// File:        matrix.c
// Description: Implementation of matrix computation
// Author:      Haris Wang
//
#include <stdlib.h>
#include <string.h>

void matrix_multiply(const float *a, const float *b, float *c, const int M, const int N, const int K)
{
    /**
     * matrix multiply
     * c = a * b
     * 
     * a    [M,N]
     * b    [N,K]
     * c    [M,K]
     * */
    register int i,j,p;
    register int a_offset=0;
    for(i=0; i<M; i++)
    {
        for(j=0; j<N; j++,a_offset++)
        {
            register float apart = a[a_offset];
            if(apart<0.00001 && apart>(0-0.00001))
                continue;
            register int c_offset = i*K;
            register int b_offset = j*K;
            for(p=0; p<K; p++)
            {
                c[c_offset++] += apart * b[b_offset++];
            }
        }
    }
}

matrix_transpose(float *x, int m, int n)
{
    /** matrix transpose
     * 
     * Input
     *      x[m,n]
     * Output
     *      x[n,m]
     * */
    float *tmp = (float *)malloc(m*n*sizeof(float));
    int i,j;
    for(i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            tmp[j*m+i] = x[i*n+j];
        }
    }
    memcpy(x, tmp, m*n*sizeof(float));
    free(tmp);
}
