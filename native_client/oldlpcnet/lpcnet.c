/* Copyright (c) 2018 Mozilla */
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

#include <math.h>
#include <stdio.h>
#include "nnet_data.h"
#include "nnet.h"
#include "common.h"
#include "arch.h"
#include "lpcnet.h"
#include "freq.h"
//#include "../CL/cl.h"

#define LPC_ORDER 16
#define PREEMPH 0.85f

#define PITCH_GAIN_FEATURE 37
#define PDF_FLOOR 0.002

#define FRAME_INPUT_SIZE (NB_FEATURES + EMBED_PITCH_OUT_SIZE)

#define SAMPLE_INPUT_SIZE (2 * EMBED_SIG_OUT_SIZE + EMBED_EXC_OUT_SIZE + FEATURE_DENSE2_OUT_SIZE)

#define FEATURES_DELAY (FEATURE_CONV1_DELAY + FEATURE_CONV2_DELAY)
struct LPCNetState
{
    NNetState nnet;
    int last_exc;
    float last_sig[LPC_ORDER];
    float old_input[FEATURES_DELAY][FEATURE_CONV2_OUT_SIZE];
    float old_lpc[FEATURES_DELAY][LPC_ORDER];
    float old_gain[FEATURES_DELAY];
    int frame_count;
    float deemph_mem;
};

#if 0
static void print_vector(float *x, int N)
{
    int i;
    for (i=0;i<N;i++) printf("%f ", x[i]);
    printf("\n");
}
#endif

void run_frame_network(LPCNetState *lpcnet, float *condition, float *gru_a_condition, const float *features, int pitch)
{
    int i;
    NNetState *net;
    float in[FRAME_INPUT_SIZE];
    float conv1_out[FEATURE_CONV1_OUT_SIZE];
    float conv2_out[FEATURE_CONV2_OUT_SIZE];
    float dense1_out[FEATURE_DENSE1_OUT_SIZE];
    if (lpcnet->frame_count == 0)
        fprintf(stderr, "FRAME_INPUT_SIZE: %d NB_FEATURES: %d EMBED_PITCH_OUT_SIZE: %d FEATURE_CONV2_OUT_SIZE: %d\n",
                FRAME_INPUT_SIZE, NB_FEATURES, EMBED_PITCH_OUT_SIZE, FEATURE_CONV2_OUT_SIZE);

    net = &lpcnet->nnet;
    RNN_COPY(in, features, NB_FEATURES);
    compute_embedding(&embed_pitch, &in[NB_FEATURES], pitch);
    if (lpcnet->frame_count == 0)
        fprintf(stderr, "pitch: %d in: %f %f\n", pitch, in[NB_FEATURES], in[NB_FEATURES + 1]);
    celt_assert(FRAME_INPUT_SIZE == feature_conv1.nb_inputs);
    compute_conv1d(&feature_conv1, conv1_out, net->feature_conv1_state, in);
    if (lpcnet->frame_count < FEATURE_CONV1_DELAY)
        RNN_CLEAR(conv1_out, FEATURE_CONV1_OUT_SIZE);
    compute_conv1d(&feature_conv2, conv2_out, net->feature_conv2_state, conv1_out);
    celt_assert(FRAME_INPUT_SIZE == FEATURE_CONV2_OUT_SIZE);
    if (lpcnet->frame_count < FEATURES_DELAY)
        RNN_CLEAR(conv2_out, FEATURE_CONV2_OUT_SIZE);
    for (i = 0; i < FEATURE_CONV2_OUT_SIZE; i++)
        conv2_out[i] += lpcnet->old_input[FEATURES_DELAY - 1][i];
    memmove(lpcnet->old_input[1], lpcnet->old_input[0], (FEATURES_DELAY - 1) * FRAME_INPUT_SIZE * sizeof(in[0]));
    memcpy(lpcnet->old_input[0], in, FRAME_INPUT_SIZE * sizeof(in[0]));
    compute_dense(&feature_dense1, dense1_out, conv2_out);
    compute_dense(&feature_dense2, condition, dense1_out);
    compute_dense(&gru_a_dense_feature, gru_a_condition, condition);
    if (lpcnet->frame_count < 1000)
        lpcnet->frame_count++;
}

void run_sample_network(NNetState *net, float *pdf, const float *condition, const float *gru_a_condition, int last_exc, int last_sig, int pred)
{
    float gru_a_input[3 * GRU_A_STATE_SIZE];
    float in_b[GRU_A_STATE_SIZE + FEATURE_DENSE2_OUT_SIZE];
    RNN_COPY(gru_a_input, gru_a_condition, 3 * GRU_A_STATE_SIZE);
    accum_embedding(&gru_a_embed_sig, gru_a_input, last_sig);
    accum_embedding(&gru_a_embed_pred, gru_a_input, pred);
    accum_embedding(&gru_a_embed_exc, gru_a_input, last_exc);
    /*compute_gru3(&gru_a, net->gru_a_state, gru_a_input);*/
    compute_sparse_gru(&sparse_gru_a, net->gru_a_state, gru_a_input);
    RNN_COPY(in_b, net->gru_a_state, GRU_A_STATE_SIZE);
    RNN_COPY(&in_b[GRU_A_STATE_SIZE], condition, FEATURE_DENSE2_OUT_SIZE);
    compute_gru2(&gru_b, net->gru_b_state, in_b);
    compute_mdense(&dual_fc, pdf, net->gru_b_state);
}

int lpcnet_get_size()
{
    return sizeof(LPCNetState);
}

int lpcnet_init(LPCNetState *lpcnet)
{
    memset(lpcnet, 0, lpcnet_get_size());
    lpcnet->last_exc = 128;
    return 0;
}

LPCNetState *lpcnet_create()
{
    LPCNetState *lpcnet;
    lpcnet = (LPCNetState *)calloc(lpcnet_get_size(), 1);
    lpcnet_init(lpcnet);
    return lpcnet;
}

void lpcnet_destroy(LPCNetState *lpcnet)
{
    free(lpcnet);
}
int factorial(int n)
{
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
const char *kernelSource =
    "__kernel void vecAdd(  __global double *a,           "
    "                    __global double *b,              "
    "                   __global double *c,           "
    "                  const unsigned int n)      "
    "{                                  "
    ""
    "    //Get our global thread ID           "
    ""
    "    int id = get_global_id(0);           "
    ""
    "                                        "
    "   //Make sure we do not go out of bounds     "
    "  if (id < n)                        "
    "      c[id] = a[id] + b[id];      "
    "}    ;";

/*void print_devices()
{
    int i, j;
    char *value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id *platforms;
    cl_uint deviceCount;
    cl_device_id *devices;
    cl_uint maxComputeUnits;

    
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (int i = 0; i < platformCount; i++)
    {

        
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

       
        for (j = 0; j < deviceCount; j++)
        {

           
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j + 1, value);
            free(value);

            
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j + 1, 1, value);
            free(value);

            
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j + 1, 2, value);
            free(value);

        
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j + 1, 3, value);
            free(value);

            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j + 1, 4, maxComputeUnits);
        }

        free(devices);
    }

    free(platforms);
}
*/

/*static char *read_source_file(const char *filename)
{
    long int
        size = 0,
        res = 0;

    char *src = NULL;

    FILE *file = fopen(filename, "rb");

    if (!file)
        return NULL;

    if (fseek(file, 0, SEEK_END))
    {
        fclose(file);
        return NULL;
    }

    size = ftell(file);
    if (size == 0)
    {
        fclose(file);
        return NULL;
    }

    rewind(file);

    src = (char *)calloc(size + 1, sizeof(char));
    if (!src)
    {
        src = NULL;
        fclose(file);
        return src;
    }

    res = fread(src, 1, sizeof(char) * size, file);
    if (res != sizeof(char) * size)
    {
        fclose(file);
        free(src);

        return src;
    }

    src[size] = '\0'; 
    fclose(file);

    return src;
}*/

/*
void excecute_open_cl()
{
    FILE *programHandle;
    size_t programSize, kernelSourceSize;
    char *programBuffer, *kernelSource;
    programHandle = fopen("kernel.cl", "rb");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char *)malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);
    printf(programBuffer);
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
    kernel = clCreateKernel(program, "vecAdd", &err);

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
}*/

int synthesize_features(char *input, char *output, int use_taco)
{
    //print_devices();
    /*for (int i = 0; i < 90000000; i++)
    {
        excecute_open_cl();
    }*/
     

    FILE *fin, *fout;
    LPCNetState *net;
    net = lpcnet_create();
    fin = fopen(input, "rb");

    fout = fopen(output, "wb");
    if (fout == NULL)
    {
        fprintf(stderr, "Can't open %s\n", output);
        exit(1);
    }

    while (1)
    {
        float features[NB_FEATURES];
        short pcm[FRAME_SIZE];
        if (use_taco == 1)
        {
            float in_features[NB_BANDS + 2];
            fread(in_features, sizeof(features[0]), NB_BANDS + 2, fin);
            if (feof(fin))
                break;
            RNN_COPY(features, in_features, NB_BANDS);
            RNN_CLEAR(&features[18], 18);
            RNN_COPY(features + 36, in_features + NB_BANDS, 2);
        }
        else
        {
            float in_features[NB_TOTAL_FEATURES];
            fread(in_features, sizeof(features[0]), NB_TOTAL_FEATURES, fin);
            if (feof(fin))
                break;
            RNN_COPY(features, in_features, NB_FEATURES);
            RNN_CLEAR(&features[18], 18);
        }
        lpcnet_synthesize(net, pcm, features, FRAME_SIZE);
        fwrite(pcm, sizeof(pcm[0]), FRAME_SIZE, fout);
    }
    fclose(fin);
    fclose(fout);
    lpcnet_destroy(net);
    return 0;
}

void lpcnet_synthesize(LPCNetState *lpcnet, short *output, const float *features, int N)
{
    int i;
    float condition[FEATURE_DENSE2_OUT_SIZE];
    float lpc[LPC_ORDER];
    float pdf[DUAL_FC_OUT_SIZE];
    float gru_a_condition[3 * GRU_A_STATE_SIZE];
    int pitch;
    float pitch_gain;
    /* FIXME: Remove this -- it's just a temporary hack to match the Python code. */
    static int start = LPC_ORDER + 1;
    /* FIXME: Do proper rounding once the Python code rounds properly. */
    pitch = (int)floor(0.1 + 50 * features[36] + 100);
    pitch_gain = lpcnet->old_gain[FEATURES_DELAY - 1];
    memmove(&lpcnet->old_gain[1], &lpcnet->old_gain[0], (FEATURES_DELAY - 1) * sizeof(lpcnet->old_gain[0]));
    lpcnet->old_gain[0] = features[PITCH_GAIN_FEATURE];
    run_frame_network(lpcnet, condition, gru_a_condition, features, pitch);
    memcpy(lpc, lpcnet->old_lpc[FEATURES_DELAY - 1], LPC_ORDER * sizeof(lpc[0]));
    memmove(lpcnet->old_lpc[1], lpcnet->old_lpc[0], (FEATURES_DELAY - 1) * LPC_ORDER * sizeof(lpc[0]));
    lpc_from_cepstrum(lpcnet->old_lpc[0], features);
    if (lpcnet->frame_count <= FEATURES_DELAY)
    {
        RNN_CLEAR(output, N);
        return;
    }
    for (i = start; i < N; i++)
    {
        int j;
        float pcm;
        int exc;
        int last_sig_ulaw;
        int pred_ulaw;
        float pred = 0;
        for (j = 0; j < LPC_ORDER; j++)
            pred -= lpcnet->last_sig[j] * lpc[j];
        last_sig_ulaw = lin2ulaw(lpcnet->last_sig[0]);
        pred_ulaw = lin2ulaw(pred);
        run_sample_network(&lpcnet->nnet, pdf, condition, gru_a_condition, lpcnet->last_exc, last_sig_ulaw, pred_ulaw);
        exc = sample_from_pdf(pdf, DUAL_FC_OUT_SIZE, MAX16(0, 1.5f * pitch_gain - .5f), PDF_FLOOR);
        pcm = pred + ulaw2lin(exc);
        RNN_MOVE(&lpcnet->last_sig[1], &lpcnet->last_sig[0], LPC_ORDER - 1);
        lpcnet->last_sig[0] = pcm;
        lpcnet->last_exc = exc;
        pcm += PREEMPH * lpcnet->deemph_mem;
        lpcnet->deemph_mem = pcm;
        if (pcm < -32767)
            pcm = -32767;
        if (pcm > 32767)
            pcm = 32767;
        output[i] = (int)floor(.5 + pcm);
    }
    start = 0;
}

#if 1

#endif
