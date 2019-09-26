/* Copyright (c) 2018 Mozilla
                 2012-2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/*
  AVX implementation of vector operations, compile with -mavx
  AVX2/FMA implementation of vector operations, compile with -mavx2 -mfma
*/

#include <immintrin.h>
#include "../CL/cl.h"

#ifdef __AVX2__
static __m256 exp8_approx(__m256 X)
{
   const __m256 K0 = _mm256_set1_ps(0.99992522f);
   const __m256 K1 = _mm256_set1_ps(0.69583354f);
   const __m256 K2 = _mm256_set1_ps(0.22606716f);
   const __m256 K3 = _mm256_set1_ps(0.078024523f);
   const __m256 log2_E = _mm256_set1_ps(1.44269504);
   const __m256 max_in = _mm256_set1_ps(50.f);
   const __m256 min_in = _mm256_set1_ps(-50.f);
   const __m256i mask = _mm256_set1_epi32(0x7fffffff);
   __m256 XF, Y;
   __m256i I;
   X = _mm256_mul_ps(X, log2_E);
   X = _mm256_max_ps(min_in, _mm256_min_ps(max_in, X));
   XF = _mm256_floor_ps(X);
   I = _mm256_cvtps_epi32(XF);
   X = _mm256_sub_ps(X, XF);
   Y = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(K3, X, K2), X, K1), X, K0);
   I = _mm256_slli_epi32(I, 23);
   Y = _mm256_castsi256_ps(_mm256_and_si256(mask, _mm256_add_epi32(I, _mm256_castps_si256(Y))));
   return Y;
}
#else
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define _mm_fmadd_ps(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
static __m128 exp4_approx(__m128 X)
{
   const __m128 K0 = _mm_set1_ps(0.99992522f);
   const __m128 K1 = _mm_set1_ps(0.69583354f);
   const __m128 K2 = _mm_set1_ps(0.22606716f);
   const __m128 K3 = _mm_set1_ps(0.078024523f);
   const __m128 log2_E = _mm_set1_ps(1.44269504);
   const __m128 max_in = _mm_set1_ps(50.f);
   const __m128 min_in = _mm_set1_ps(-50.f);
   const __m128i mask = _mm_set1_epi32(0x7fffffff);
   __m128 XF, Y;
   __m128i I;
   X = _mm_mul_ps(X, log2_E);
   X = _mm_max_ps(min_in, _mm_min_ps(max_in, X));
   XF = _mm_floor_ps(X);
   I = _mm_cvtps_epi32(XF);
   X = _mm_sub_ps(X, XF);
   Y = _mm_fmadd_ps(_mm_fmadd_ps(_mm_fmadd_ps(K3, X, K2), X, K1), X, K0);
   I = _mm_slli_epi32(I, 23);
   Y = _mm_castsi128_ps(_mm_and_si128(mask, _mm_add_epi32(I, _mm_castps_si128(Y))));
   return Y;
}
static __m256 exp8_approx(__m256 X)
{
   __m256 Y;
   __m128 Xhi, Xlo, Yhi, Ylo;
   Xhi = _mm256_extractf128_ps(X, 1);
   Xlo = _mm256_extractf128_ps(X, 0);
   Yhi = exp4_approx(Xhi);
   Ylo = exp4_approx(Xlo);
   Y = _mm256_insertf128_ps(_mm256_setzero_ps(), Yhi, 1);
   Y = _mm256_insertf128_ps(Y, Ylo, 0);
   return Y;
}
#endif

static float celt_exp(float x)
{
   float out[8];
   __m256 X, Y;
   X = _mm256_set1_ps(x);
   Y = exp8_approx(X);
   _mm256_storeu_ps(out, Y);
   return out[0];
}

static void softmax(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N - 7; i += 8)
   {
      __m256 X, Y;
      X = _mm256_loadu_ps(&x[i]);
      Y = exp8_approx(X);
      _mm256_storeu_ps(&y[i], Y);
   }
   for (; i < N; i++)
      y[i] = celt_exp(x[i]);
}

static void vec_tanh(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N - 7; i += 8)
   {
      const __m256 two = _mm256_set1_ps(2.f);
      const __m256 one = _mm256_set1_ps(1.f);
      __m256 X, Y;
      X = _mm256_loadu_ps(&x[i]);
      X = _mm256_mul_ps(X, two);
      Y = exp8_approx(X);
      Y = _mm256_mul_ps(_mm256_sub_ps(Y, one), _mm256_rcp_ps(_mm256_add_ps(Y, one)));
      _mm256_storeu_ps(&y[i], Y);
   }
   for (; i < N; i++)
   {
      float ex2;
      ex2 = celt_exp(2 * x[i]);
      y[i] = (ex2 - 1) / (ex2 + 1);
   }
}

static void vec_sigmoid(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N - 7; i += 8)
   {
      const __m256 one = _mm256_set1_ps(1.f);
      __m256 X, Y;
      X = _mm256_loadu_ps(&x[i]);
      Y = exp8_approx(X);
      Y = _mm256_mul_ps(Y, _mm256_rcp_ps(_mm256_add_ps(Y, one)));
      _mm256_storeu_ps(&y[i], Y);
   }
   for (; i < N; i++)
   {
      float ex;
      ex = celt_exp(x[i]);
      y[i] = (ex) / (ex + 1);
   }
}

static void sgemv_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{
   int i, j;
   for (i = 0; i < rows; i += 16)
   {
      float *__restrict y;
      __m256 vy0, vy8;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      for (j = 0; j < cols; j++)
      {
         __m256 vxj;
         __m256 vw;
         vxj = _mm256_broadcast_ss(&x[j]);

         vw = _mm256_loadu_ps(&weights[j * col_stride + i]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vw = _mm256_loadu_ps(&weights[j * col_stride + i + 8]);
         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
      }
      _mm256_storeu_ps(&y[0], vy0);
      _mm256_storeu_ps(&y[8], vy8);
   }
}
static void sparse_sgemv_accum16(float *out, const float *weights, int rows, const int *idx, const float *x)
{
  /* int value_id_in_idx, cols_num = rows / 16;

   int *cols_list; //cols_num
   cols_list = (int *)malloc(sizeof(int) * cols_num);

   int *idx_id_list; //cols_num + 1
   idx_id_list = (int *)malloc(sizeof(int) * (cols_num + 1));

   value_id_in_idx = idx_id_list[0] = 1;
   for (int i = 0; i < cols_num; i++)
   {
      value_id_in_idx = idx_id_list[i];
      cols_list[i] = idx[value_id_in_idx - 1];
      idx_id_list[i + 1] = cols_list[i] + idx_id_list[i] + 1;
   }

   int weights_id = 0;
   int *weights_id_list;
   weights_id_list = (int *)malloc(sizeof(int) * (rows));
   for (int i = 0; i < rows / 16; i++)
   {
      weights_id_list[i] = weights_id;
      weights_id += cols_list[i] * 16;
   }
#pragma loop(hint_parallel(4))
#pragma loop( ivdep )
   for (int i = 0; i < rows; i += 16)
   {
      int list_id = i / 16;
      int cols = cols_list[list_id];
      int idx_start_id = idx_id_list[list_id];

      float *y = &out[i];
      __m256 vy0 = _mm256_loadu_ps(&y[0]);
      __m256 vy8 = _mm256_loadu_ps(&y[8]);

      for (int j = 0; j < cols; j++)
      {
         __m256 vxj, vw;

         int id = idx[idx_start_id + j];
         int weights_id = j * 16 + weights_id_list[list_id];

         vxj = _mm256_broadcast_ss(&x[id]);
         vw = _mm256_loadu_ps(&weights[weights_id]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);
         vw = _mm256_loadu_ps(&weights[weights_id + 8]);
         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
      }

      _mm256_storeu_ps(&y[0], vy0);
      _mm256_storeu_ps(&y[8], vy8);
   }

   free(cols_list);

   free(idx_id_list);

   free(weights_id_list);
*/

    int i, j;
   for (i = 0; i < rows; i += 16)
   {
      float *__restrict y;
      int cols;
      __m256 vy0, vy8;
      y = &out[i];
      vy0 = _mm256_loadu_ps(&y[0]);
      vy8 = _mm256_loadu_ps(&y[8]);
      cols = *idx++;
      for (j = 0; j < cols; j++)
      {
         int id;
         __m256 vxj;
         __m256 vw;
         id = *idx++;
         vxj = _mm256_broadcast_ss(&x[id]);

         vw = _mm256_loadu_ps(&weights[0]);
         vy0 = _mm256_fmadd_ps(vw, vxj, vy0);

         vw = _mm256_loadu_ps(&weights[8]);
         vy8 = _mm256_fmadd_ps(vw, vxj, vy8);
         weights += 16;
      }
      _mm256_storeu_ps(&y[0], vy0);
      _mm256_storeu_ps(&y[8], vy8);
   }
}