#!/bin/bash

# V100: 80 SMs, 4 warp schedulers/SM, max 2048 threads/SM
# don't forget to set the BUILD_DIR env variable

threads_per_tb_copy=1024  # max_threads_per_sm / 2 = 2048 / 2
threads_per_tb_comp=128   # 4 warps (1 per scheduler subpartition)
num_itrs_comp=20000000    # adjust to match isolated copy runtime
num_itrs_copy=40          # adjust to match isolated compute runtime
num_bytes=2147483648      # 2 GB (conservative for V100 16GB/32GB)

ILP=4

# Measure the latency of a single copy kernel
$BUILD_DIR/ipc 1 0 $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes

# Measure the latency of a single compute kernel
$BUILD_DIR/ipc 1 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes

# Measure the latency of a copy and compute kernel running sequentially
# NOTE: update num_itrs_comp and num_itrs_copy such that their isolated runtime is similar
$BUILD_DIR/ipc 2 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes

# Measure the latency of a copy and compute kernel running concurrently
$BUILD_DIR/ipc 3 $ILP $threads_per_tb_copy $threads_per_tb_comp $num_itrs_copy $num_itrs_comp $num_bytes