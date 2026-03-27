#!/bin/bash

# NOTE: current arguments are tailored to the H100
# adapt them based on the GPU you are running on
# don't forget to set the BUILD_DIR env variable

num_tb=40 # V100: half of 80 SMs
num_threads_per_tb=1024
num_itrs=10000

# V100 L2 cache = 6MB, sweep from 1MB to 6MB in 1MB steps
# for (NUM_BYTES = 1MB; NUM_BYTES <= 6MB; NUM_BYTES += 1MB)
for NUM_BYTES in $(seq 1048576 1048576 6291456); do
    # Measure latency of single copy kernel
    $BUILD_DIR/l2_cache 1 $num_tb $num_threads_per_tb $num_itrs $NUM_BYTES

    # Measure latency of two colocated copy kernel
    $BUILD_DIR/l2_cache 3 $num_tb $num_threads_per_tb $num_itrs $NUM_BYTES
done;
