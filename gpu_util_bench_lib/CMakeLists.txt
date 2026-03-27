add_library(gpu_util_bench kernels.cu)

target_include_directories(gpu_util_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpu_util_bench PUBLIC nvidia-ml)

# Enable PIC so this static lib can be linked into shared libraries (python_interface.so)
set_target_properties(gpu_util_bench PROPERTIES POSITION_INDEPENDENT_CODE ON)