#!/bin/bash

# V100: 80 SMs, 4 warp schedulers/SM
# num_threads_per_tb = num_schedulers_per_sm * 32 = 4 * 32 = 128 (1 warp per subpartition)
# don't forget to set the BUILD_DIR env variable

num_threads_per_tb=128 # 1 warp per SM subpartition (same as H100)
num_itrs=20000000


for ILP in $(seq 1 1 4); do
    # run two compute kernels sequentially
    $BUILD_DIR/pipelines 2 $ILP $num_threads_per_tb $num_itrs

    # run two compute kernels colocated using CUDA streams
    $BUILD_DIR/pipelines 3 $ILP $num_threads_per_tb $num_itrs
done