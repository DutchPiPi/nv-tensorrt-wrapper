/*
* Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
 */
 
/**
 * @file
 * TensorRT wrapper header for dnn_backend in ffmpeg.
 */

#ifndef TRT_CLASS_WRAPPER_H
#define TRT_CLASS_WRAPPER_H

#ifdef __cplusplus
extern "C"
{
#endif

    #include "../dnn_interface.h"

    #define NUM_TRT_IO 2

    typedef struct TRTOptions
    {
        int device;
        char *plugin_so;
        int stream;
    } TRTOptions;
    
    typedef struct TRTContext
    {
        const AVClass *av_class;

        TRTOptions options;
        AVBufferRef *hwdevice;
        // Device memory pointer to the fp32 CHW input/output of the model
        // The device memory is only allocated once and reused during inference
        // Multiple input/output is not supported
        void *trt_in, *trt_out;
        // Device memory pointer to 8-bit image data
        void *frame_in, *frame_out;

        int channels, packed;
    } TRTContext;
    
    typedef int tloadModelTrt(DNNModel *model, TRTContext *ctx, const char *model_filename);
    typedef int texecuteModelTrt(const DNNModel *model, uint8_t **in_frame_data, int *in_linesize,
                                int in_width, int in_height, uint8_t **out_frame_data, int *out_linesize,
                                int out_width, int out_height, int packed, cudaStream_t stream);
    typedef void tfreeModelTrt(DNNModel *model);

    typedef struct TRTWrapper
    {
        void *so_handle;
        tloadModelTrt *load_model_func;
        texecuteModelTrt *execute_model_func;
        tfreeModelTrt *free_model_func;

        TRTContext *ctx;
    } TRTWrapper;

#ifdef __cplusplus
}
#endif
#endif
