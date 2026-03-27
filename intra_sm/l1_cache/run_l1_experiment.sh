#!/bin/bash

# V100: 80 SMs, 4 warp schedulers/SM, unified L1 = 128 KB
# don't forget to set the BUILD_DIR env variable

num_threads_per_tb=64 # num_schedulers_per_sm(4) * 32 / 2 = 64
unified_l1_cache_size=128 # V100 unified L1 cache (including shared memory) = 128 KB
num_itrs=15000

# V100 unified L1 = 128KB, sweep from 16KB to 64KB in 16KB steps
# for (num_bytes_per_tb = 16KB; num_bytes_per_tb <= 64KB; num_bytes_per_tb += 16KB)
for num_bytes_per_tb in $(seq 16384 16384 65536); do
    # measure sequential latency of two copy kernel
    $BUILD_DIR/l1_cache 2 $num_threads_per_tb $num_bytes_per_tb $unified_l1_cache_size $num_itrs

    # measure concurrent latency of two copy kernels
    $BUILD_DIR/l1_cache 3 $num_threads_per_tb $num_bytes_per_tb $unified_l1_cache_size $num_itrs
done;