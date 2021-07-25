# nv-tensorrt-wrapper

C interfaces for TensorRT C++ API. 

Run:

`make && make install`

to build and install the wrapper before building FFmpeg with vf_tensorrt. By default, the wrapper will be installed under /usr/local/lib and /usr/local/include, which might require sudo privilege, you can modify the install prefix by

`make install PREFIX=<path>`

You can also specify CUDA path and TensorRT install path

`make install CUDA_PATH=<cuda path> TRT_PATH=<trt path>`