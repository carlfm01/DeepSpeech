/* Copyright (c) 2018 Mozilla
                 2008-2011 Octasic Inc.
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
/* No AVX2/FMA support */
#include "../CL/cl.h"
#include <stdio.h>
#ifndef LPCNET_TEST


static float celt_exp2(float x)
{
   int integer;
   float frac;
   union {
      float f;
      opus_uint32 i;
   } res;
   integer = floor(x);
   if (integer < -50)
      return 0;
   frac = x - integer;
   /* K0 = 1, K1 = log(2), K2 = 3-4*log(2), K3 = 3*log(2) - 2 */
   res.f = 0.99992522f + frac * (0.69583354f + frac * (0.22606716f + 0.078024523f * frac));
   res.i = (res.i + (integer << 23)) & 0x7fffffff;
   return res.f;
}
#define celt_exp(x) celt_exp2((x)*1.44269504f)

static float tansig_approx(float x)
{
   int i;
   float y, dy;
   float sign = 1;
   /* Tests are reversed to catch NaNs */
   if (!(x < 8))
      return 1;
   if (!(x > -8))
      return -1;
#ifndef FIXED_POINT
   /* Another check in case of -ffast-math */
   if (celt_isnan(x))
      return 0;
#endif
   if (x < 0)
   {
      x = -x;
      sign = -1;
   }
   i = (int)floor(.5f + 25 * x);
   x -= .04f * i;
   y = tansig_table[i];
   dy = 1 - y * y;
   y = y + x * dy * (1 - y * x);
   return sign * y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
   return .5f + .5f * tansig_approx(.5f * x);
}

static void softmax(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N; i++)
      y[i] = celt_exp(x[i]);
}

static void vec_tanh(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N; i++)
   {
      y[i] = tansig_approx(x[i]);
   }
}

static void vec_sigmoid(float *y, const float *x, int N)
{
   int i;
   for (i = 0; i < N; i++)
   {
      y[i] = sigmoid_approx(x[i]);
   }
}
#endif



void run_gemm()
{
   printf("Running OpenCL on ");
    FILE *programHandle;
    size_t programSize, kernelSourceSize;
    char *programBuffer, *kernelSource;
    programHandle = fopen("gemm.cl", "rb");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char *)malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);
    //printf(programBuffer);
    // Length of vectors
    unsigned int n = 10000000;

    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;

    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;

    cl_platform_id cpPlatform; // OpenCL platform
    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double *)malloc(bytes);
    h_b = (double *)malloc(bytes);
    h_c = (double *)malloc(bytes);

    // Initialize vectors on host
    int i;
    for (i = 0; i < n; i++)
    {
        h_a[i] = sinf(i) * sinf(i);
        h_b[i] = cosf(i) * cosf(i);
    }

    size_t globalSize, localSize;
    cl_int err;

    // Number of work items in each local work group
    localSize = 128;

    // Number of total work items - localSize must be devisor
    globalSize = ceil(n / (float)localSize) * localSize;
    //printf(globalSize);
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);

    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&programBuffer, &programSize, &err);

    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "sgemv_accum16", &err);

    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                               bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                bytes, h_b, 0, NULL, NULL);

    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                 0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                        bytes, h_c, 0, NULL, NULL);

    //Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/n);

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
}

static void sgemv_accum16(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{

  // run_gemm();
   /*int i, j;
   for (i = 0; i < rows; i += 16)
   {
     // printf("Account +=16 %d\n", i);
      //printf(i);
      for (j = 0; j < cols; j++)
      {
         //printf("Account j %d\n", j);
         const float *   w;
         float *  y;
         float xj;
         w = &weights[j * col_stride + i];
         xj = x[j];
         y = &out[i];
         y[0] += w[0] * xj;
         y[1] += w[1] * xj;
         y[2] += w[2] * xj;
         y[3] += w[3] * xj;
         y[4] += w[4] * xj;
         y[5] += w[5] * xj;
         y[6] += w[6] * xj;
         y[7] += w[7] * xj;
         y[8] += w[8] * xj;
         y[9] += w[9] * xj;
         y[10] += w[10] * xj;
         y[11] += w[11] * xj;
         y[12] += w[12] * xj;
         y[13] += w[13] * xj;
         y[14] += w[14] * xj;
         y[15] += w[15] * xj;
      }
   }*/
   int t,j;
   int loops = rows / 16;
   
   for ( t = 0; t < loops; t++)
   {
      int i = t * 16;
      //printf("Account +=16 %d\n", i);
      //printf(i);
      for ( j = 0; j < cols; j++)
      {
         //printf("Account j %d\n", j);
         const float * __restrict  w;
         float * __restrict y;
         float xj;
         w = &weights[j * col_stride + i];
         xj = x[j];
         y = &out[i];
         y[0] += w[0] * xj;
         y[1] += w[1] * xj;
         y[2] += w[2] * xj;
         y[3] += w[3] * xj;
         y[4] += w[4] * xj;
         y[5] += w[5] * xj;
         y[6] += w[6] * xj;
         y[7] += w[7] * xj;
         y[8] += w[8] * xj;
         y[9] += w[9] * xj;
         y[10] += w[10] * xj;
         y[11] += w[11] * xj;
         y[12] += w[12] * xj;
         y[13] += w[13] * xj;
         y[14] += w[14] * xj;
         y[15] += w[15] * xj;
      }
   }
}

static void sparse_sgemv_accum16(float *out, const float *w, int rows, const int *idx, const float *x)
{
   //run_gemm();
   int i, j;
   for (i=0;i<rows;i+=16)
   {
      int cols;
      cols = *idx++;
      for (j=0;j<cols;j++)
      {
         float * __restrict y;
         float xj;
         xj = x[*idx++];
         y = &out[i];
         y[0] += w[0]*xj;
         y[1] += w[1]*xj;
         y[2] += w[2]*xj;
         y[3] += w[3]*xj;
         y[4] += w[4]*xj;
         y[5] += w[5]*xj;
         y[6] += w[6]*xj;
         y[7] += w[7]*xj;
         y[8] += w[8]*xj;
         y[9] += w[9]*xj;
         y[10] += w[10]*xj;
         y[11] += w[11]*xj;
         y[12] += w[12]*xj;
         y[13] += w[13]*xj;
         y[14] += w[14]*xj;
         y[15] += w[15]*xj;
         w += 16;
      }
   }
}



