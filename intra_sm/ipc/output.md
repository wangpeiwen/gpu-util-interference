------------------------------------
Running copy kernel alone - Launch Config Copy: (80, 1024)
Alone time is 333.669 ms
Alone time is 326.812 ms
Alone time is 326.367 ms
Alone time is 326.102 ms
Alone time is 325.524 ms
Alone time is 326.677 ms
Alone time is 326.517 ms
Alone time is 325.691 ms
Alone time is 327.792 ms
Alone time is 324.719 ms
Avg alone time is 326.442 ms
------------------------------------
Running compute kernel with ILP 4 alone - Launch Config Compute: (80, 128)
Alone time is 125.708 ms
Alone time is 125.257 ms
Alone time is 125.251 ms
Alone time is 125.25 ms
Alone time is 125.249 ms
Alone time is 125.251 ms
Alone time is 125.25 ms
Alone time is 125.278 ms
Alone time is 125.253 ms
Alone time is 125.25 ms
Avg alone time is 125.251 ms
------------------------------------
Running copy and compute with ILP 4 sequentially - Launch Config Copy: (80, 1024) Launch Config Compute: (80, 128)
Sequential time is 457.754 ms
Sequential time is 451.129 ms
Sequential time is 450.601 ms
Sequential time is 451.47 ms
Sequential time is 451.206 ms
Sequential time is 453.269 ms
Sequential time is 450.772 ms
Sequential time is 451.869 ms
Sequential time is 453.059 ms
Sequential time is 451.262 ms
Avg sequential time is 451.366 ms
------------------------------------
Running copy and compute with ILP 4 colocated - Launch Config Copy: (80, 1024) Launch Config Compute: (80, 128)
Colocated time is 327.017 ms
Colocated time is 329.105 ms
Colocated time is 325.861 ms
Colocated time is 327.814 ms
Colocated time is 326.095 ms
Colocated time is 326.208 ms
Colocated time is 328.088 ms
Colocated time is 327.576 ms
Colocated time is 326.642 ms
Colocated time is 327.913 ms
Avg colocated time is 327.297 ms