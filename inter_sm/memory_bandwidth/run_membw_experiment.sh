#!/bin/bash

# V100: 80 SMs, max 2048 threads/SM, L2 = 6 MB, 16/32 GB HBM2
# MPS 50% -> 40 SMs available per process
# Each block: 1024 threads -> 1 block/SM (1024 < 2048)
# Full wave = 40 blocks; loop 1~4 waves

NUM_THREADS_PER_BLOCK=1024  # max_threads_per_sm / 2 = 2048 / 2
NUM_ITRS=50
NUM_BYTES=2147483648  # 2 GB (safe for 16 GB V100 with two processes)

echo "------------------------------------"
echo "Starting MPS"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

function cleanup() {
    echo "Shutting down MPS control daemon..."
    echo quit | nvidia-cuda-mps-control
    echo "MPS control daemon shut down."
}

trap "cleanup; exit" SIGINT

# V100: 40 available SMs with 50% MPS, step by full waves (40 blocks)
for NUM_TB in 40 80 120 160; do
    # measuring latency of a single copy kernel on 50% of SMs
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 $BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES

    # measuring latencies of two copy kernels colocated each running on 50% of SMs
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 $BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES &
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 $BUILD_DIR/mem_bw 1 $NUM_TB $NUM_THREADS_PER_BLOCK $NUM_ITRS $NUM_BYTES
done

cleanup
