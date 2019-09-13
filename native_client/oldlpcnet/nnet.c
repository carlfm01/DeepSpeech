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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <math.h>
#include "opus_types.h"
#include "arch.h"
#include "common.h"
#include "tansig_table.h"
#include "nnet.h"
#include "nnet_data.h"
#include "stdint.h"
#include "../CL/cl.h"
#include <stdio.h>
#define SOFTMAX_HACK

#ifdef __AVX__
#include "vec_avx.h"
#elif __ARM_NEON__
#include "vec_neon.h"
#else
#include "vec.h"
#endif
void checkErrorCl_Status(cl_int error, int line)
{

   if (error != CL_SUCCESS)
   {
      printf("------------------------------------------------------------------------------------------------");
      switch (error)
      {
      case CL_DEVICE_NOT_FOUND:
         printf("-- Error at %d:  Device not found.\n", line);
         break;
      case CL_DEVICE_NOT_AVAILABLE:
         printf("-- Error at %d:  Device not available\n", line);
         break;
      case CL_COMPILER_NOT_AVAILABLE:
         printf("-- Error at %d:  Compiler not available\n", line);
         break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:
         printf("-- Error at %d:  Memory object allocation failure\n", line);
         break;
      case CL_OUT_OF_RESOURCES:
         printf("-- Error at %d:  Out of resources\n", line);
         break;
      case CL_OUT_OF_HOST_MEMORY:
         printf("-- Error at %d:  Out of host memory\n", line);
         break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:
         printf("-- Error at %d:  Profiling information not available\n", line);
         break;
      case CL_MEM_COPY_OVERLAP:
         printf("-- Error at %d:  Memory copy overlap\n", line);
         break;
      case CL_IMAGE_FORMAT_MISMATCH:
         printf("-- Error at %d:  Image format mismatch\n", line);
         break;
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:
         printf("-- Error at %d:  Image format not supported\n", line);
         break;
      case CL_BUILD_PROGRAM_FAILURE:
         printf("-- Error at %d:  Program build failure\n", line);
         break;
      case CL_MAP_FAILURE:
         printf("-- Error at %d:  Map failure\n", line);
         break;
      case CL_INVALID_VALUE:
         printf("-- Error at %d:  Invalid value\n", line);
         break;
      case CL_INVALID_DEVICE_TYPE:
         printf("-- Error at %d:  Invalid device type\n", line);
         break;
      case CL_INVALID_PLATFORM:
         printf("-- Error at %d:  Invalid platform\n", line);
         break;
      case CL_INVALID_DEVICE:
         printf("-- Error at %d:  Invalid device\n", line);
         break;
      case CL_INVALID_CONTEXT:
         printf("-- Error at %d:  Invalid context\n", line);
         break;
      case CL_INVALID_QUEUE_PROPERTIES:
         printf("-- Error at %d:  Invalid queue properties\n", line);
         break;
      case CL_INVALID_COMMAND_QUEUE:
         printf("-- Error at %d:  Invalid command queue\n", line);
         break;
      case CL_INVALID_HOST_PTR:
         printf("-- Error at %d:  Invalid host pointer\n", line);
         break;
      case CL_INVALID_MEM_OBJECT:
         printf("-- Error at %d:  Invalid memory object\n", line);
         break;
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
         printf("-- Error at %d:  Invalid image format descriptor\n", line);
         break;
      case CL_INVALID_IMAGE_SIZE:
         printf("-- Error at %d:  Invalid image size\n", line);
         break;
      case CL_INVALID_SAMPLER:
         printf("-- Error at %d:  Invalid sampler\n", line);
         break;
      case CL_INVALID_BINARY:
         printf("-- Error at %d:  Invalid binary\n", line);
         break;
      case CL_INVALID_BUILD_OPTIONS:
         printf("-- Error at %d:  Invalid build options\n", line);
         break;
      case CL_INVALID_PROGRAM:
         printf("-- Error at %d:  Invalid program\n", line);
         break;
      case CL_INVALID_PROGRAM_EXECUTABLE:
         printf("-- Error at %d:  Invalid program executable\n", line);
         break;
      case CL_INVALID_KERNEL_NAME:
         printf("-- Error at %d:  Invalid kernel name\n", line);
         break;
      case CL_INVALID_KERNEL_DEFINITION:
         printf("-- Error at %d:  Invalid kernel definition\n", line);
         break;
      case CL_INVALID_KERNEL:
         printf("-- Error at %d:  Invalid kernel\n", line);
         break;
      case CL_INVALID_ARG_INDEX:
         printf("-- Error at %d:  Invalid argument index\n", line);
         break;
      case CL_INVALID_ARG_VALUE:
         printf("-- Error at %d:  Invalid argument value\n", line);
         break;
      case CL_INVALID_ARG_SIZE:
         printf("-- Error at %d:  Invalid argument size\n", line);
         break;
      case CL_INVALID_KERNEL_ARGS:
         printf("-- Error at %d:  Invalid kernel arguments\n", line);
         break;
      case CL_INVALID_WORK_DIMENSION:
         printf("-- Error at %d:  Invalid work dimensionsension\n", line);
         break;
      case CL_INVALID_WORK_GROUP_SIZE:
         printf("-- Error at %d:  Invalid work group size\n", line);
         break;
      case CL_INVALID_WORK_ITEM_SIZE:
         printf("-- Error at %d:  Invalid work item size\n", line);
         break;
      case CL_INVALID_GLOBAL_OFFSET:
         printf("-- Error at %d:  Invalid global offset\n", line);
         break;
      case CL_INVALID_EVENT_WAIT_LIST:
         printf("-- Error at %d:  Invalid event wait list\n", line);
         break;
      case CL_INVALID_EVENT:
         printf("-- Error at %d:  Invalid event\n", line);
         break;
      case CL_INVALID_OPERATION:
         printf("-- Error at %d:  Invalid operation\n", line);
         break;
      case CL_INVALID_GL_OBJECT:
         printf("-- Error at %d:  Invalid OpenGL object\n", line);
         break;
      case CL_INVALID_BUFFER_SIZE:
         printf("-- Error at %d:  Invalid buffer size\n", line);
         break;
      case CL_INVALID_MIP_LEVEL:
         printf("-- Error at %d:  Invalid mip-map level\n", line);
         break;
      case -1024:
         printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line);
         break;
      case -1023:
         printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line);
         break;
      case -1022:
         printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line);
         break;
      case -1021:
         printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line);
         break;
      case -1020:
         printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line);
         break;
      case -1019:
         printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line);
         break;
      case -1018:
         printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line);
         break;
      case -1017:
         printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line);
         break;
      case -1016:
         printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line);
         break;
      case -1015:
         printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line);
         break;
      case -1014:
         printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line);
         break;
      case -1013:
         printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line);
         break;
      case -1012:
         printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line);
         break;
      case -1011:
         printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line);
         break;
      case -1010:
         printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line);
         break;
      case -1009:
         printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line);
         break;
      case -1008:
         printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line);
         break;
      case -1007:
         printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line);
         break;
      case -1001:
         printf("-- Error at %d:  Code -1001: no GPU available?\n", line);
         break;
      default:
         printf("-- Error at %d:  Unknown with code %d\n", line, error);
      }
   }
}
static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}

void printTest()
{
   const char *source =
       "__kernel void main(__global int2 *out) {\n"
       "      out[get_global_id(0)]++;\n"
       "}\n";
   cl_command_queue command_queue;
   cl_context context;
   cl_device_id device;
   cl_int input[] = {0, 1, 2, 3};
   const size_t global_work_size = sizeof(input) / sizeof(cl_int2);
   cl_kernel kernel;
   cl_mem buffer;
   cl_platform_id platform;
   cl_program program;

   clGetPlatformIDs(1, &platform, NULL);
   clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
   command_queue = clCreateCommandQueue(context, device, 0, NULL);
   buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(input), &input, NULL);
   program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
   clBuildProgram(program, 1, &device, "", NULL, NULL);
   kernel = clCreateKernel(program, "main", NULL);
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
   clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
   clFlush(command_queue);
   clFinish(command_queue);
   clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, 0, sizeof(input), &input, 0, NULL, NULL);

   printf("1|%d\n", input[0]);
   printf("2|%d\n", input[1]);
   printf("3|%d\n", input[2]);
   printf("4|%d\n", input[3]);
}

static void sgemv_accum(float *out, const float *weights, int rows, int cols, int col_stride, const float *x)
{

   //printf("length of output rows |%d\n", sizeof(rows));
   //printf("length of output cols |%d\n", sizeof(cols));
 int i, j;
   if (rows % 16 == 0)
   {
      sgemv_accum16(out, weights, rows, cols, col_stride, x);
   }
   else
   {
      for (i = 0; i < rows; i++)
      {
         //printf("|%d", cols);
         for (j = 0; j < cols; j++)
            out[i] = (weights[j * col_stride + i] * x[j]) + out[i];
      }
   }


   //printTest();

   //matrix 1
   //rows = M
   //cols = K
   
  /* FILE *programHandle;
   size_t programSize, kernelSourceSize;
   char *programBuffer, *kernelSource;
   programHandle = fopen("sgemmaccum.cl", "rb");
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

   size_t out_bytes = sizeof(float) * rows;
   size_t x_bytes = sizeof(float) * cols;
   size_t weights_bytes = sizeof(float) * (cols * col_stride + rows);
   size_t rows_bytes = sizeof(int);
   size_t cols_bytes = sizeof(int);
   size_t col_stride_bytes = sizeof(int);
   // Host output vector
   float *h_neuron_output;

   // Device input buffers
   cl_mem d_neuron_out;
   cl_mem d_weights;
   //cl_mem d_rows;
   //cl_mem d_cols;
   //cl_mem c_computed_out;
   cl_mem d_x;
   // Device output buffer
   cl_mem d_c;

   cl_platform_id cpPlatform; // OpenCL platform
   cl_device_id device_id;    // device ID
   cl_context context;        // context
   cl_command_queue queue;    // command queue
   cl_program program;        // program
   cl_kernel kernel;          // kernel

   // Allocate memory for each vector on host
   h_neuron_output = (float *)malloc(out_bytes);

   size_t globalSize, localSize;
   cl_int err;

   // Number of work items in each local work group
   localSize = 32;

   // Number of total work items - localSize must be devisor
   globalSize = ceil(rows / (float)localSize) * localSize;

   //printf(globalSize);
   // Bind to platform
   err = clGetPlatformIDs(1, &cpPlatform, &err);
   checkErrorCl_Status(err, __LINE__);
   // Get ID for the device
   err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
   checkErrorCl_Status(err, __LINE__);
   // Create a context
   context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
   checkErrorCl_Status(err, __LINE__);
   // Create a command queue
   queue = clCreateCommandQueue(context, device_id, 0, &err);
   checkErrorCl_Status(err, __LINE__);
   // Create the compute program from the source buffer
   program = clCreateProgramWithSource(context, 1,
                                       (const char **)&programBuffer, &programSize, &err);
   checkErrorCl_Status(err, __LINE__);

   // Build the program executable
   clBuildProgram(program, 0, NULL, NULL, NULL, &err);
   checkErrorCl_Status(err, __LINE__);
   // Create the compute kernel in the program we wish to run
   kernel = clCreateKernel(program, "sgemv_accum", &err);
   checkErrorCl_Status(err, __LINE__);

   // Create the input and output arrays in device memory for our calculation
   d_neuron_out = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (void *)out_bytes, out, &err);
   checkErrorCl_Status(err, __LINE__);
   d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (void *)weights_bytes, weights, &err);
   checkErrorCl_Status(err, __LINE__);
   //d_rows = clCreateBuffer(context, CL_MEM_READ_ONLY, 4, NULL, NULL);
   //d_cols = clCreateBuffer(context, CL_MEM_READ_ONLY, 4, NULL, NULL);
   //d_stride = clCreateBuffer(context, CL_MEM_READ_ONLY, 4, NULL, NULL);
   d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (void *)x_bytes, x, &err);
   checkErrorCl_Status(err, __LINE__);
   d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (void *)out_bytes, NULL, &err);
   checkErrorCl_Status(err, __LINE__);

   // Write our data set into the input array in device memory
   //float *out, const float *weights, int rows, int cols, int col_stride, const float *x
   err = clEnqueueWriteBuffer(queue, d_neuron_out, CL_TRUE, 0,
                              out_bytes, out, 0, NULL, NULL);
   checkErrorCl_Status(err, __LINE__);
   err |= clEnqueueWriteBuffer(queue, d_weights, CL_TRUE, 0,
                               weights_bytes, weights, 0, NULL, NULL);
   checkErrorCl_Status(err, __LINE__);
   //err |= clEnqueueWriteBuffer(queue, d_rows, CL_TRUE, 0,
   //                           4, rows, 0, NULL, NULL);
   //err |= clEnqueueWriteBuffer(queue, d_cols, CL_TRUE, 0,
   //                           4, cols, 0, NULL, NULL);
   //err |= clEnqueueWriteBuffer(queue, d_stride, CL_TRUE, 0,
   //                           4, col_stride, 0, NULL, NULL);
   err |= clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0,
                               x_bytes, x, 0, NULL, NULL);

   checkErrorCl_Status(err, __LINE__);

   //err |= clEnqueueWriteBuffer(queue, d_c, CL_TRUE, 0,
   //                            sizeof(float)*rows, h_neuron_output, 0, NULL, NULL);

   checkErrorCl_Status(err, __LINE__);

   // Set the arguments to our compute kernel
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_neuron_out);
   checkErrorCl_Status(err, __LINE__);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_weights);
   checkErrorCl_Status(err, __LINE__);
   err |= clSetKernelArg(kernel, 2, sizeof(int), (void *)&rows);
   checkErrorCl_Status(err, __LINE__);
   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&cols);
   checkErrorCl_Status(err, __LINE__);
   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&col_stride);
   checkErrorCl_Status(err, __LINE__);
   err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&d_x);
   checkErrorCl_Status(err, __LINE__);
   err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&d_c);
   checkErrorCl_Status(err, __LINE__);
   // Execute the kernel over the entire range of the data set
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                0, NULL, &err);
   checkErrorCl_Status(err, __LINE__);
   //printf("Row:: %d\n", rows);
   // Wait for the command queue to get serviced before reading back results
   clFinish(queue);

   // Read the results from the device
   err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                             (void *)out_bytes, h_neuron_output, 0, NULL, &err);

   checkErrorCl_Status(err, __LINE__);

  /* for (int i = 0; i < rows; i++)
   {
      out[i] = h_neuron_output[i];
   }

   //Sum up vector c and print result divided by n, this should equal 1 within error
   //double sum = 0;
*/
  
   /*
   for (int pp = 0; pp < rows; pp++)
   {
      printf("final result gt: %f\n", out[pp]);
      printf("final result: %f\n", h_neuron_output[pp]);
   }*/
   //out=h_neuron_output;
   //printf("final result: %f\n", sum / n);

   // release OpenCL resources*/
  /* clReleaseMemObject(d_neuron_out);
   clReleaseMemObject(d_weights);
   clReleaseMemObject(d_x);
   // Device output buffer
   clReleaseMemObject(d_c);

   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseContext(context);

   //release host memory
   //free(h_a);
   // free(h_b);
   free(h_neuron_output);*/
}

void compute_activation(float *output, float *input, int N, int activation)
{
   int i;
   if (activation == ACTIVATION_SIGMOID)
   {
      vec_sigmoid(output, input, N);
   }
   else if (activation == ACTIVATION_TANH)
   {
      vec_tanh(output, input, N);
   }
   else if (activation == ACTIVATION_RELU)
   {
      for (i = 0; i < N; i++)
         output[i] = relu(input[i]);
   }
   else if (activation == ACTIVATION_SOFTMAX)
   {
#ifdef SOFTMAX_HACK
      for (i = 0; i < N; i++)
         output[i] = input[i];
#else
      float sum = 0;
      softmax(output, input, N);
      for (i = 0; i < N; i++)
      {
         sum += output[i];
      }
      sum = 1.f / (sum + 1e-30);
      for (i = 0; i < N; i++)
         output[i] = sum * output[i];
#endif
   }
   else
   {
      celt_assert(activation == ACTIVATION_LINEAR);
      for (i = 0; i < N; i++)
         output[i] = input[i];
   }
}

void compute_dense(const DenseLayer *layer, float *output, const float *input)
{
   int i;
   int N, M;
   int stride;
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   stride = N;
   celt_assert(input != output);
   for (i = 0; i < N; i++)
      output[i] = layer->bias[i];
   sgemv_accum(output, layer->input_weights, N, M, stride, input);
   compute_activation(output, output, N, layer->activation);
}

void compute_mdense(const MDenseLayer *layer, float *output, const float *input)
{
   int i, c;
   int N, M, C;
   int stride;
   float tmp[MAX_MDENSE_TMP];
   celt_assert(input != output);
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   C = layer->nb_channels;
   celt_assert(N * C <= MAX_MDENSE_TMP);
   stride = N * C;
   for (i = 0; i < N * C; i++)
      tmp[i] = layer->bias[i];
   sgemv_accum(tmp, layer->input_weights, N * C, M, stride, input);
   compute_activation(tmp, tmp, N * C, ACTIVATION_TANH);
   for (i = 0; i < N; i++)
      output[i] = 0;
   for (c = 0; c < C; c++)
   {
      for (i = 0; i < N; i++)
         output[i] += tmp[c * N + i] * layer->factor[c * N + i];
   }
   compute_activation(output, output, N, layer->activation);
}

void compute_gru(const GRULayer *gru, float *state, const float *input)
{
   int i;
   int N, M;
   int stride;
   float tmp[MAX_RNN_NEURONS];
   float z[MAX_RNN_NEURONS];
   float r[MAX_RNN_NEURONS];
   float h[MAX_RNN_NEURONS];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   stride = 3 * N;
   /* Compute update gate. */
   for (i = 0; i < N; i++)
      z[i] = gru->bias[i];
   if (gru->reset_after)
   {
      for (i = 0; i < N; i++)
         z[i] += gru->bias[3 * N + i];
   }
   sgemv_accum(z, gru->input_weights, N, M, stride, input);
   sgemv_accum(z, gru->recurrent_weights, N, N, stride, state);
   compute_activation(z, z, N, ACTIVATION_SIGMOID);

   /* Compute reset gate. */
   for (i = 0; i < N; i++)
      r[i] = gru->bias[N + i];
   if (gru->reset_after)
   {
      for (i = 0; i < N; i++)
         r[i] += gru->bias[4 * N + i];
   }
   sgemv_accum(r, &gru->input_weights[N], N, M, stride, input);
   sgemv_accum(r, &gru->recurrent_weights[N], N, N, stride, state);
   compute_activation(r, r, N, ACTIVATION_SIGMOID);

   /* Compute output. */
   for (i = 0; i < N; i++)
      h[i] = gru->bias[2 * N + i];
   if (gru->reset_after)
   {
      for (i = 0; i < N; i++)
         tmp[i] = gru->bias[5 * N + i];
      sgemv_accum(tmp, &gru->recurrent_weights[2 * N], N, N, stride, state);
      for (i = 0; i < N; i++)
         h[i] += tmp[i] * r[i];
      sgemv_accum(h, &gru->input_weights[2 * N], N, M, stride, input);
   }
   else
   {
      for (i = 0; i < N; i++)
         tmp[i] = state[i] * r[i];
      sgemv_accum(h, &gru->input_weights[2 * N], N, M, stride, input);
      sgemv_accum(h, &gru->recurrent_weights[2 * N], N, N, stride, tmp);
   }
   compute_activation(h, h, N, gru->activation);
   for (i = 0; i < N; i++)
      h[i] = z[i] * state[i] + (1 - z[i]) * h[i];
   for (i = 0; i < N; i++)
      state[i] = h[i];
}

void compute_gru2(const GRULayer *gru, float *state, const float *input)
{
   int i;
   int N, M;
   int stride;
   float zrh[3 * MAX_RNN_NEURONS];
   float recur[3 * MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2 * N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   stride = 3 * N;
   /* Compute update gate. */
   for (i = 0; i < 3 * N; i++)
      zrh[i] = gru->bias[i];
   sgemv_accum(zrh, gru->input_weights, 3 * N, M, stride, input);
   for (i = 0; i < 3 * N; i++)
      recur[i] = gru->bias[3 * N + i];
   sgemv_accum(recur, gru->recurrent_weights, 3 * N, N, stride, state);
   for (i = 0; i < 2 * N; i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, 2 * N, ACTIVATION_SIGMOID);
   for (i = 0; i < N; i++)
      h[i] += recur[2 * N + i] * r[i];
   compute_activation(h, h, N, gru->activation);
   for (i = 0; i < N; i++)
      h[i] = z[i] * state[i] + (1 - z[i]) * h[i];
   for (i = 0; i < N; i++)
      state[i] = h[i];
}

void compute_gru3(const GRULayer *gru, float *state, const float *input)
{
   int i;
   int N;
   int stride;
   float zrh[3 * MAX_RNN_NEURONS];
   float recur[3 * MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2 * N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   stride = 3 * N;
   RNN_COPY(zrh, input, 3 * N);
   for (i = 0; i < 3 * N; i++)
      recur[i] = gru->bias[3 * N + i];
   sgemv_accum(recur, gru->recurrent_weights, 3 * N, N, stride, state);
   for (i = 0; i < 2 * N; i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, 2 * N, ACTIVATION_SIGMOID);
   for (i = 0; i < N; i++)
      h[i] += recur[2 * N + i] * r[i];
   compute_activation(h, h, N, gru->activation);
   for (i = 0; i < N; i++)
      h[i] = z[i] * state[i] + (1 - z[i]) * h[i];
   for (i = 0; i < N; i++)
      state[i] = h[i];
}

void compute_sparse_gru(const SparseGRULayer *gru, float *state, const float *input)
{
   int i, k;
   int N;
   float zrh[3 * MAX_RNN_NEURONS];
   float recur[3 * MAX_RNN_NEURONS];
   float *z;
   float *r;
   float *h;
   N = gru->nb_neurons;
   z = zrh;
   r = &zrh[N];
   h = &zrh[2 * N];
   celt_assert(gru->nb_neurons <= MAX_RNN_NEURONS);
   celt_assert(input != state);
   celt_assert(gru->reset_after);
   RNN_COPY(zrh, input, 3 * N);
   for (i = 0; i < 3 * N; i++)
      recur[i] = gru->bias[3 * N + i];
   for (k = 0; k < 3; k++)
   {
      for (i = 0; i < N; i++)
         recur[k * N + i] += gru->diag_weights[k * N + i] * state[i];
   }
   sparse_sgemv_accum16(recur, gru->recurrent_weights, 3 * N, gru->idx, state);
   for (i = 0; i < 2 * N; i++)
      zrh[i] += recur[i];
   compute_activation(zrh, zrh, 2 * N, ACTIVATION_SIGMOID);
   for (i = 0; i < N; i++)
      h[i] += recur[2 * N + i] * r[i];
   compute_activation(h, h, N, gru->activation);
   for (i = 0; i < N; i++)
      h[i] = z[i] * state[i] + (1 - z[i]) * h[i];
   for (i = 0; i < N; i++)
      state[i] = h[i];
}

void compute_conv1d(const Conv1DLayer *layer, float *output, float *mem, const float *input)
{
   int i;
   int N, M;
   int stride;
   float tmp[MAX_CONV_INPUTS];
   celt_assert(input != output);
   celt_assert(layer->nb_inputs * layer->kernel_size <= MAX_CONV_INPUTS);
   RNN_COPY(tmp, mem, layer->nb_inputs * (layer->kernel_size - 1));
   RNN_COPY(&tmp[layer->nb_inputs * (layer->kernel_size - 1)], input, layer->nb_inputs);
   M = layer->nb_inputs * layer->kernel_size;
   N = layer->nb_neurons;
   stride = N;
   for (i = 0; i < N; i++)
      output[i] = layer->bias[i];
   sgemv_accum(output, layer->input_weights, N, M, stride, tmp);
   compute_activation(output, output, N, layer->activation);
   RNN_COPY(mem, &tmp[layer->nb_inputs], layer->nb_inputs * (layer->kernel_size - 1));
}

void compute_embedding(const EmbeddingLayer *layer, float *output, int input)
{
   int i;
   celt_assert(input >= 0);
   celt_assert(input < layer->nb_inputs);
   /*if (layer->dim == 64) printf("%d\n", input);*/
   for (i = 0; i < layer->dim; i++)
   {
      output[i] = layer->embedding_weights[input * layer->dim + i];
   }
}

void accum_embedding(const EmbeddingLayer *layer, float *output, int input)
{
   int i;
   celt_assert(input >= 0);
   celt_assert(input < layer->nb_inputs);
   /*if (layer->dim == 64) printf("%d\n", input);*/
   for (i = 0; i < layer->dim; i++)
   {
      output[i] += layer->embedding_weights[input * layer->dim + i];
   }
}
/* needed to replace Windows/gcc rand() with our own rand() function
   to get click free synthesised audio - not sure why */
#define NNET_RAND_MAX 32768
static uint32_t next = 1;
uint16_t nnet_rand(void)
{
   next = next * 1103515245 + 12345;
   uint32_t r = (next / 65536) % 32768;
   return ((uint16_t)r);
}

int sample_from_pdf(const float *pdf, int N, float exp_boost, float pdf_floor)
{
   int i;
   float sum, norm;
   float r;
   float tmp[DUAL_FC_OUT_SIZE];
   celt_assert(N <= DUAL_FC_OUT_SIZE);
   sum = 0;
#ifdef SOFTMAX_HACK
   for (i = 0; i < N; i++)
   {
      tmp[i] = pdf[i] * (1.f + exp_boost);
   }
   softmax(tmp, tmp, N);
   for (i = 0; i < N; i++)
   {
      sum += tmp[i];
   }
#else
   /* Decrease the temperature of the sampling. */
   for (i = 0; i < N; i++)
   {
      tmp[i] = pow(pdf[i], 1.f + exp_boost);
      sum += tmp[i];
   }
#endif
   norm = 1.f / sum;
   /* Convert tmp to a CDF while subtracting the floor */
   tmp[0] = MAX16(0, norm * tmp[0] - pdf_floor);
   for (i = 1; i < N; i++)
   {
      tmp[i] = tmp[i - 1] + MAX16(0, norm * tmp[i] - pdf_floor);
   }
   /* Do the sampling (from the cdf). */
   float annr = (float)nnet_rand();
   float arand = (annr / NNET_RAND_MAX);
   r = tmp[N - 1] * arand;

   for (i = 0; i < N - 1; i++)
   {
      if (r < tmp[i])
         return i;
   }
   //fprintf(stderr, "DUAL_FC_OUT_SIZE: %d annr: %f arand: %f\n", DUAL_FC_OUT_SIZE, annr, arand);
   return N - 1;
}
