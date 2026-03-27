#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include "kernels.cuh"

using namespace std;

#define CUDACHECK(cmd)                                                                            \
    do                                                                                            \
    {                                                                                             \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess)                                                                     \
        {                                                                                         \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

__global__ void fma_fp32_ilp4(float *a, float *b, float *c, long long num_itrs)
{
    float op1 = a[threadIdx.x];
    float op2 = b[threadIdx.x];
    float op3 = 0.0f;
    float op4 = 0.0f;
    float op5 = 0.0f;
    float op6 = 0.0f;
    for (long long i = 0; i < num_itrs; i++)
    {
        op3 = __fmaf_rn(op1, op2, op3);
        op4 = __fmaf_rn(op1, op2, op4);
        op5 = __fmaf_rn(op1, op2, op5);
        op6 = __fmaf_rn(op1, op2, op6);
    }
    c[threadIdx.x] = op3 + op4 + op5 + op6;
}

template <typename T>
void init_compute_memory(T **a, T **b, T **c, T **d_a, T **d_b, T **d_c, int num_elems, cudaStream_t stream)
{
    size_t size_bytes = num_elems * sizeof(T);

    *a = (T *)malloc(size_bytes);
    *b = (T *)malloc(size_bytes);
    *c = (T *)malloc(size_bytes);

    CUDACHECK(cudaMalloc(d_a, size_bytes));
    CUDACHECK(cudaMalloc(d_b, size_bytes));
    CUDACHECK(cudaMalloc(d_c, size_bytes));

    for (int i = 0; i < num_elems; i++)
    {
        (*a)[i] = 1.0f;
        (*b)[i] = 1.0f;
    }

    CUDACHECK(cudaMemcpyAsync(*d_a, *a, size_bytes, cudaMemcpyHostToDevice, stream));
    CUDACHECK(cudaMemcpyAsync(*d_b, *b, size_bytes, cudaMemcpyHostToDevice, stream));
}

template <typename T>
void free_compute_memory(T **a, T **b, T **c, T **d_a, T **d_b, T **d_c)
{
    free(*a);
    free(*b);
    free(*c);
    CUDACHECK(cudaFree(*d_a));
    CUDACHECK(cudaFree(*d_b));
    CUDACHECK(cudaFree(*d_c));
}

extern "C" void run_fp32_fma_kernel(int num_tb, int num_threads_per_tb, long long num_itrs, int num_runs)
{
    float *a, *b, *c, *d_a, *d_b, *d_c;

    // launch in non default stream in order to overlap with mm.py kernel
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    init_compute_memory<float>(&a, &b, &c, &d_a, &d_b, &d_c, num_threads_per_tb, stream);

    // create events to record latency of each run
    vector<cudaEvent_t> start_events(num_runs);
    vector<cudaEvent_t> stop_events(num_runs);
    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventCreate(&start_events[i]));
        CUDACHECK(cudaEventCreate(&stop_events[i]));
    }

    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventRecord(start_events[i], stream));
        fma_fp32_ilp4<<<num_tb, num_threads_per_tb, 0, stream>>>(d_a, d_b, d_c, num_itrs);
        CUDACHECK(cudaEventRecord(stop_events[i], stream));
    }

    CUDACHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < num_runs; i++)
    {
        float latency;
        CUDACHECK(cudaEventElapsedTime(&latency, start_events[i], stop_events[i]));
        printf("[FP32] Run %d: Latency: %f ms\n", i, latency);

        CUDACHECK(cudaEventDestroy(start_events[i]));
        CUDACHECK(cudaEventDestroy(stop_events[i]));
    }

    CUDACHECK(cudaStreamSynchronize(stream));
    free_compute_memory<float>(&a, &b, &c, &d_a, &d_b, &d_c);

    printf("FP32 Kernel execution completed!\n");
}

// ── Block Scheduler 压力核 ──
// 使用 sleep_kernel 高频 launch 占满 Block Scheduler 队列
extern "C" void run_tb_scheduler_stress(int num_tb, int num_threads_per_tb, long long num_itrs, int num_runs)
{
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    vector<cudaEvent_t> start_events(num_runs);
    vector<cudaEvent_t> stop_events(num_runs);
    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventCreate(&start_events[i]));
        CUDACHECK(cudaEventCreate(&stop_events[i]));
    }

    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventRecord(start_events[i], stream));
        sleep_kernel<<<num_tb, num_threads_per_tb, 0, stream>>>(num_itrs);
        CUDACHECK(cudaEventRecord(stop_events[i], stream));
    }

    CUDACHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < num_runs; i++)
    {
        float latency;
        CUDACHECK(cudaEventElapsedTime(&latency, start_events[i], stop_events[i]));
        printf("[BS Stress] Run %d: Latency: %f ms\n", i, latency);
        CUDACHECK(cudaEventDestroy(start_events[i]));
        CUDACHECK(cudaEventDestroy(stop_events[i]));
    }

    CUDACHECK(cudaStreamDestroy(stream));
    printf("Block Scheduler stress kernel completed!\n");
}

// ── L2 Cache 压力核 ──
// 使用 copy_kernel + L2 大小工作集制造 cache thrashing
extern "C" void run_l2_cache_stress(int num_tb, int num_threads_per_tb, long long num_itrs, long long num_bytes, int num_runs)
{
    long long num_floats = num_bytes / sizeof(float);
    float *d_in, *d_out;
    CUDACHECK(cudaMalloc(&d_in, num_bytes));
    CUDACHECK(cudaMalloc(&d_out, num_bytes));
    CUDACHECK(cudaMemset(d_in, 0, num_bytes));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    vector<cudaEvent_t> start_events(num_runs);
    vector<cudaEvent_t> stop_events(num_runs);
    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventCreate(&start_events[i]));
        CUDACHECK(cudaEventCreate(&stop_events[i]));
    }

    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventRecord(start_events[i], stream));
        copy_kernel<<<num_tb, num_threads_per_tb, 0, stream>>>(d_in, d_out, num_floats, num_itrs);
        CUDACHECK(cudaEventRecord(stop_events[i], stream));
    }

    CUDACHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < num_runs; i++)
    {
        float latency;
        CUDACHECK(cudaEventElapsedTime(&latency, start_events[i], stop_events[i]));
        printf("[L2 Stress] Run %d: Latency: %f ms\n", i, latency);
        CUDACHECK(cudaEventDestroy(start_events[i]));
        CUDACHECK(cudaEventDestroy(stop_events[i]));
    }

    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(d_in));
    CUDACHECK(cudaFree(d_out));
    printf("L2 Cache stress kernel completed!\n");
}

// ── Memory Bandwidth 压力核 ──
// 使用 copy_kernel + 大工作集占满显存带宽
extern "C" void run_mem_bw_stress(int num_tb, int num_threads_per_tb, long long num_itrs, long long num_bytes, int num_runs)
{
    long long num_floats = num_bytes / sizeof(float);
    float *d_in, *d_out;
    CUDACHECK(cudaMalloc(&d_in, num_bytes));
    CUDACHECK(cudaMalloc(&d_out, num_bytes));
    CUDACHECK(cudaMemset(d_in, 0, num_bytes));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    vector<cudaEvent_t> start_events(num_runs);
    vector<cudaEvent_t> stop_events(num_runs);
    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventCreate(&start_events[i]));
        CUDACHECK(cudaEventCreate(&stop_events[i]));
    }

    for (int i = 0; i < num_runs; i++)
    {
        CUDACHECK(cudaEventRecord(start_events[i], stream));
        copy_kernel<<<num_tb, num_threads_per_tb, 0, stream>>>(d_in, d_out, num_floats, num_itrs);
        CUDACHECK(cudaEventRecord(stop_events[i], stream));
    }

    CUDACHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < num_runs; i++)
    {
        float latency;
        CUDACHECK(cudaEventElapsedTime(&latency, start_events[i], stop_events[i]));
        printf("[BW Stress] Run %d: Latency: %f ms\n", i, latency);
        CUDACHECK(cudaEventDestroy(start_events[i]));
        CUDACHECK(cudaEventDestroy(stop_events[i]));
    }

    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaFree(d_in));
    CUDACHECK(cudaFree(d_out));
    printf("Memory Bandwidth stress kernel completed!\n");
}