//
// File:        matrix.c
// Description: Implementation of matrix computation
// Author:      Haris Wang
//
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

/*
// 
// SIMD
//
void matrix_multiply(const float *a, const float *b, float *c, const int M, const int N, const int K)
{
    register int i,j;
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
            __m128 zero={};
            while(c_offset%4!=0)
            {
                c_offset--;
            } 
            while(b_offset%4!=0)
            {
                b_offset--;
            } 
            while(c_offset<(i+1)*K-4)
            {
                __m128  ma=zero+apart;  
                __m128  mb;  
                __m128  mc;  
                mb = _mm_load_ps(b+b_offset);  
                mc = _mm_load_ps(c+c_offset);           
                mc = _mm_add_ps(mc, _mm_mul_ps(ma, mb));
                _mm_store_ps(c+c_offset, mc); 
                c_offset+=4;
                b_offset+=4;
            }
            while(c_offset<(i+1)*K)
            {
                c[c_offset++] += apart * b[b_offset++];
            }
        }
    }
}
*/

//
// Todo: Explore more efficient matrix_multiply algorithm
//
void matrix_multiply(const float *a, const float *b, float *c, const int M, const int N, const int K)
{
    /**
     * matrix multiply, c = a * b
     * 
     * Input:
     * a    [M,N]
     * b    [N,K]
     * Output:
     * c    [M,K]
     * */
    register int i,j,p;
    register float *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        register float *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register float apart = *(a_ptr++);
            if (apart<0.00001 && apart>(0-0.00001))
                continue;
            register float *c_ptr = c + i*K;
            for (p = 0; p < K; p++)
                *(c_ptr++) += *(b_ptr++) * apart;
        }
    }
}

matrix_transpose(float *x, int m, int n)
{
    /** matrix transpose
     * 
     * Input:
     *      x[m,n]
     * Output:
     *      x[n,m]
     * */
    float *tmp = (float *)malloc(m*n*sizeof(float));
    register int i, j;
    register float *ptr = x;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            tmp[j*m+i] = *(ptr++);
    }
    memcpy(x, tmp, m*n*sizeof(float));
    free(tmp);
}
