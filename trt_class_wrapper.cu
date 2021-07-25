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
 * DNN TensorRT backend C++ wrapper.
 */

#include <cuda_runtime.h>

#include <bits/stdint-uintn.h>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <sstream>
#include <mutex>

#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define SOCKET int
#define INVALID_SOCKET -1

#include <NvInfer.h>

#include "trt_class_wrapper.h"
#include "Logger.h"

using namespace nvinfer1;
using namespace std;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define BLOCKX 32
#define BLOCKY 16

extern simplelogger::Logger *logger;
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);

// TensorRT section
// Self-defined CUDA check functions as cuda_check.h is not available for cpp due to void* function pointers

inline bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        printf("CUDA runtime API error: %s, at line %d in file %s\n",
            cudaGetErrorName(e), iLine, szFile);
        return false;
    }
    return true;
}

inline bool check(bool bSuccess, int iLine, const char *szFile) {
    if (!bSuccess) {
        printf("Error at line %d in file %s\n", iLine, szFile);
        return false;
    }
    return true;
}

#define ck_cu(call) check(call, __LINE__, __FILE__)

inline std::string to_string(nvinfer1::Dims const &dim) {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < dim.nbDims; i++) {
        oss << dim.d[i] << ", ";
    }
    oss << ")";
    return oss.str();
}

typedef ICudaEngine *(*BuildEngineProcType)(IBuilder *builder, void *pData);

struct IOInfo {
    string name;
    bool bInput;
    nvinfer1::Dims dim;
    nvinfer1::DataType dataType;

    string GetDimString() {
        return ::to_string(dim);
    }
    string GetDataTypeString() {
        static string aTypeName[] = {"float", "half", "int8", "int32", "bool"};
        return aTypeName[(int)dataType];
    }
    size_t GetNumBytes() {
        static int aSize[] = {4, 2, 1, 4, 1};
        size_t nSize = aSize[(int)dataType];
        for (int i = 0; i < dim.nbDims; i++) {
            nSize *= dim.d[i];
        }
        return nSize;
    }
    string to_string() {
        ostringstream oss;
        oss << setw(6) << (bInput ? "input" : "output")
            << " | " << setw(5) << GetDataTypeString()
            << " | " << GetDimString()
            << " | " << "size=" << GetNumBytes()
            << " | " << name;
        return oss.str();
    }
};

struct BuildEngineParam {
    int nMaxBatchSize;
    int nChannel, nHeight, nWidth;
    std::size_t nMaxWorkspaceSize;
    bool bFp16, bInt8, bRefit;
};

class BufferedFileReader {
public:
    BufferedFileReader(const char *szFileName, bool bPartial = false) {
        struct stat st;

        if (stat(szFileName, &st) != 0) {
            LOG(WARNING) << "File " << szFileName << " does not exist.";
            return;
        }

        nSize = st.st_size;
        while (true) {
            try {
                pBuf = new uint8_t[nSize + 1];
                if (nSize != st.st_size) {
                    LOG(WARNING) << "File is too large - only " << std::setprecision(4) << 100.0 * nSize / (uint32_t)st.st_size << "% is loaded";
                }
                break;
            } catch(std::bad_alloc&) {
                if (!bPartial) {
                    LOG(ERROR) << "Failed to allocate memory in BufferedReader";
                    return;
                }
                nSize = (uint32_t)(nSize * 0.9);
            }
        }

        FILE *fp = fopen(szFileName, "rb");
        size_t nRead = fread(pBuf, 1, nSize, fp);
        pBuf[nSize] = 0;
        fclose(fp);

        if (nRead != nSize) {
        	LOG(ERROR) << "nRead != nSize";
        }
    }
    ~BufferedFileReader() {
        if (pBuf) {
            delete[] pBuf;
        }
    }
    bool GetBuffer(uint8_t **ppBuf, uint32_t *pnSize) {
        if (!pBuf) {
            return false;
        }

        if (ppBuf) *ppBuf = pBuf;
        if (pnSize) *pnSize = nSize;
        return true;
    }

private:
    uint8_t *pBuf = NULL;
    uint32_t nSize = 0;
};

class TrtLite {
public:
    TrtLite(const char *szEnginePath, TRTContext *trt_ctx, simplelogger::Logger *logger):ctx(trt_ctx),trtLogger(logger) {
        uint8_t *pBuf = nullptr;
        uint32_t nSize = 0;
        BufferedFileReader reader(szEnginePath);

        reader.GetBuffer(&pBuf, &nSize);
        IRuntime *runtime = createInferRuntime(trtLogger);
        engine = runtime->deserializeCudaEngine(pBuf, nSize);
        runtime->destroy();
        if (!engine) {
            LOG(ERROR) << "No engine generated";
            return;
        }
    }
    virtual ~TrtLite() {
        if (context) {
            context->destroy();
        }
        if (engine) {
            engine->destroy();
        }
    }
    ICudaEngine *GetEngine() {
        return engine;
    }
    void Execute(int nBatch, vector<void *> &vdpBuf, cudaStream_t stm = 0, cudaEvent_t* evtInputConsumed = nullptr) {
        if (!engine) {
            LOG(ERROR) << "No engine";
            return;
        }
        if (!engine->hasImplicitBatchDimension() && nBatch > 1) {
            LOG(WARNING) << "Engine was built with explicit batch but is executed with batch size != 1. Results may be incorrect.";
            return;
        }
        if (engine->getNbBindings() != NUM_TRT_IO) {
            LOG(ERROR) << "Number of bindings conflicts with input and output";
            return;
        }
        if (!context) {
            context = engine->createExecutionContext();
            if (!context) {
                LOG(ERROR) << "createExecutionContext() failed";
                return;
            }
        }
        ck_cu(context->enqueue(nBatch, vdpBuf.data(), stm, evtInputConsumed));
    }
    void Execute(map<int, Dims> i2shape, vector<void *> &vdpBuf, cudaStream_t stm = 0, cudaEvent_t* evtInputConsumed = nullptr) {
        if (!engine) {
            LOG(ERROR) << "No engine";
            return;
        }
        if (engine->hasImplicitBatchDimension()) {
            LOG(ERROR) << "Engine was built with static-shaped input";
            return;
        }
        if (engine->getNbBindings() != NUM_TRT_IO) {
            LOG(ERROR) << "Number of bindings conflicts with input and output";
            return;
        }
        if (!context) {
            context = engine->createExecutionContext();
            if (!context) {
                LOG(ERROR) << "createExecutionContext() failed";
                return;
            }
        }
        for (auto &it : i2shape) {
            context->setBindingDimensions(it.first, it.second);
        }
        ck_cu(context->enqueueV2(vdpBuf.data(), stm, evtInputConsumed));
    }

    vector<IOInfo> ConfigIO(int nBatchSize) {
        vector<IOInfo> vInfo;
        if (!engine) {
            LOG(ERROR) << "No engine";
            return vInfo;
        }
        if (!engine->hasImplicitBatchDimension()) {
            LOG(ERROR) << "Engine must be built with implicit batch size (and static shape)";
            return vInfo;
        }
        for (int i = 0; i < engine->getNbBindings(); i++) {
            vInfo.push_back({string(engine->getBindingName(i)), engine->bindingIsInput(i),
                MakeDim(nBatchSize, engine->getBindingDimensions(i)), engine->getBindingDataType(i)});
        }
        return vInfo;
    }
    vector<IOInfo> ConfigIO(map<int, Dims> i2shape) {
        vector<IOInfo> vInfo;
        if (!engine) {
            LOG(ERROR) << "No engine";
            return vInfo;
        }
        if (engine->hasImplicitBatchDimension()) {
            LOG(ERROR) << "Engine must be built with explicit batch size (to enable dynamic shape)";
            return vInfo;
        }
        if (!context) {
            context = engine->createExecutionContext();
            if (!context) {
                LOG(ERROR) << "createExecutionContext() failed";
                return vInfo;
            }
        }
        for (auto &it : i2shape) {
            context->setBindingDimensions(it.first, it.second);
        }
        if (!context->allInputDimensionsSpecified()) {
            LOG(ERROR) << "Not all binding shape are specified";
            return vInfo;
        }
        for (int i = 0; i < engine->getNbBindings(); i++) {
            vInfo.push_back({string(engine->getBindingName(i)), engine->bindingIsInput(i),
                context->getBindingDimensions(i), engine->getBindingDataType(i)});
        }
        return vInfo;
    }

    void PrintInfo() {
        if (!engine) {
            LOG(ERROR) << "No engine";
            return;
        }
        LOG(INFO) << "nbBindings: " << engine->getNbBindings() << endl;
        // Only contains engine-level IO information: if dynamic shape is used,
        // dimension -1 will be printed
        for (int i = 0; i < engine->getNbBindings(); i++) {
            LOG(INFO) << "#" << i << ": " << IOInfo{string(engine->getBindingName(i)), engine->bindingIsInput(i),
                engine->getBindingDimensions(i), engine->getBindingDataType(i)}.to_string() << endl;
        }
    }
    
    TRTContext *ctx = nullptr;

private:
    static size_t GetBytesOfBinding(int iBinding, ICudaEngine *engine, IExecutionContext *context = nullptr) {
        size_t aValueSize[] = {4, 2, 1, 4, 1};
        size_t nSize = aValueSize[(int)engine->getBindingDataType(iBinding)];
        const Dims &dims = context ? context->getBindingDimensions(iBinding) : engine->getBindingDimensions(iBinding);
        for (int i = 0; i < dims.nbDims; i++) {
            nSize *= dims.d[i];
        }
        return nSize;
    }
    static nvinfer1::Dims MakeDim(int nBatchSize, nvinfer1::Dims dim) {
        nvinfer1::Dims ret(dim);
        for (int i = ret.nbDims; i > 0; i--) {
            ret.d[i] = ret.d[i - 1];
        }
        ret.d[0] = nBatchSize;
        ret.nbDims++;
        return ret;
    }

    class TrtLogger : public nvinfer1::ILogger {
    public:
        TrtLogger(simplelogger::Logger *logger) : logger(logger) {}
        void log(Severity severity, const char* msg) noexcept override {
            static simplelogger::LogLevel map[] = {
                simplelogger::FATAL, simplelogger::ERROR, simplelogger::WARNING, simplelogger::INFO, simplelogger::TRACE
            };
            simplelogger::LogTransaction(logger, map[(int)severity], __FILE__, __LINE__, __FUNCTION__).GetStream() << msg;
        }
    private:
        simplelogger::Logger *logger;
    } trtLogger;

    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
};
// End of TensorRT section

static __global__ void frame_to_dnn_kernel(uint8_t *src, int src_linesize, float *dst, int dst_linesize,
                             int width, int height, int unpack_rgb)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (unpack_rgb)
    {
        uchar3 rgb = *((uchar3 *)(src + y * src_linesize) + x);
        dst[y * dst_linesize + x] = (float)rgb.x;
        dst[y * dst_linesize + x + dst_linesize * height] = (float)rgb.y;
        dst[y * dst_linesize + x + 2 * dst_linesize * height] = (float)rgb.z;
    }
    else
    {
        dst[y * dst_linesize + x] = (float)src[y * src_linesize + x];
    }
}

static __device__ float clamp(float x, float lower, float upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

static __global__ void dnn_to_frame_kernel(float *src, int src_linesize, uint8_t *dst, int dst_linesize,
                            int width, int height, int pack_rgb)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (pack_rgb)
    {
        uint8_t r = (uint8_t)clamp(src[y * src_linesize + x], .0f, 255.0f);
        uint8_t g = (uint8_t)clamp(src[y * src_linesize + x + src_linesize * height], .0f, 255.0f);
        uint8_t b = (uint8_t)clamp(src[y * src_linesize + x + 2 * src_linesize * height], .0f, 255.0f);

        uchar3 rgb = make_uchar3(r, g, b);

        *((uchar3*)(dst + y * dst_linesize) + x) = rgb;
    }
    else
    {
        dst[y * dst_linesize + x] = (uint8_t)clamp(src[y * src_linesize + x], .0f, 255.0f);
    }
}

#define BATCH 1

#ifdef __cplusplus
extern "C"
{
#endif
#undef CTX
static DNNReturnType frame_to_dnn(uint8_t **in_frame_data, int *linesize, int width, int height,
                                    TRTContext *ctx, int num_bytes, cudaStream_t stream, int packed)
{
    ck_cu(cudaMemcpy2DAsync(ctx->frame_in, linesize[0], in_frame_data[0], linesize[0],
                            linesize[0], height, cudaMemcpyHostToDevice, stream));
    frame_to_dnn_kernel<<<dim3(DIV_UP(width, BLOCKX), DIV_UP(height, BLOCKY)), dim3(BLOCKX, BLOCKY), 0, stream>>>
                        (*(uint8_t**)&ctx->frame_in, linesize[0], *(float**)&ctx->trt_in, width, width, height, packed);

    return DNN_SUCCESS;
}

static DNNReturnType dnn_to_frame(uint8_t **out_frame_data, int *linesize, int width, int height,
                                    TRTContext *ctx, int num_bytes, cudaStream_t stream, int packed)
{
    dnn_to_frame_kernel<<<dim3(DIV_UP(width, BLOCKX), DIV_UP(height, BLOCKY)), dim3(BLOCKX, BLOCKY), 0, stream>>>
                        (*(float**)&ctx->trt_out, width, *(uint8_t**)&ctx->frame_out, linesize[0], width, height, packed);

    ck_cu(cudaMemcpy2DAsync(out_frame_data[0], linesize[0], ctx->frame_out, linesize[0],
                            linesize[0], height, cudaMemcpyDeviceToHost, stream));

    ck_cu(cudaStreamSynchronize(stream));

    return DNN_SUCCESS;
}

static DNNReturnType get_input_trt(void *model, DNNData *input, const char *input_name)
{
    TrtLite* trt_model = (TrtLite*)model;
    TRTContext *ctx = trt_model->ctx;

    LOG(INFO) << "Get TRT input.";

    // For dynamic shape, input dimensions are set to -1,
    // trt input is initialized in get_output_trt() along with trt output
    if (!trt_model->GetEngine()->hasImplicitBatchDimension())
    {
        LOG(INFO) << "Model supports dynamic shape";
        for (int i = 0; i < trt_model->GetEngine()->getNbBindings(); i++) {
            if (trt_model->GetEngine()->bindingIsInput(i))
            {
                ctx->channels = trt_model->GetEngine()->getBindingDimensions(i).d[1];
                if (ctx->channels == -1)
                {
                    LOG(ERROR) << "Do not support dynamic channel size";
                    return DNN_ERROR;
                }
                input->channels = ctx->channels;
            }
        }
        input->height = -1;
        input->width = -1;
        input->dt = DNN_FLOAT;

        return DNN_SUCCESS;
    }

    vector<IOInfo> v_info = trt_model->ConfigIO(BATCH);
    for (auto info: v_info)
    {
        if (info.bInput)
        {
            input->channels = info.dim.d[1];
            input->height = info.dim.d[2];
            input->width = info.dim.d[3];
            input->dt = DNN_FLOAT;

            ck_cu(cudaMalloc((void**)&ctx->trt_in, info.GetNumBytes()));
            ck_cu(cudaMalloc((void**)&ctx->frame_in, info.GetNumBytes() / sizeof(float)));
            #ifdef CTX
            ck_cu(cuCtxPopCurrent(&dummy));
            #endif

            return DNN_SUCCESS;
        }
    }

    LOG(ERROR) << "No input found in the model";
    return DNN_ERROR;
}

static DNNReturnType get_output_trt(void *model, const char *input_name, int input_width, int input_height,
                                const char *output_name, int *output_width, int *output_height)
{
    TrtLite* trt_model = (TrtLite*)model;
    TRTContext *ctx = trt_model->ctx;
    extern char dnn_io_proc_trt_ptx[];

    LOG(INFO) << "Get TRT output.";

    vector<IOInfo> v_info;
    if (!trt_model->GetEngine()->hasImplicitBatchDimension())
    {
        map<int, Dims> i2shape;
        i2shape.insert(make_pair(0, Dims{4, {BATCH, ctx->channels, input_height, input_width}}));
        v_info = trt_model->ConfigIO(i2shape);
    }
    else
    {
        v_info = trt_model->ConfigIO(BATCH);
    }

    for (auto info: v_info)
    {
        // For dynamic shape, inputs are initialized here
        if (info.bInput && (!trt_model->GetEngine()->hasImplicitBatchDimension()))
        {
            ck_cu(cudaMalloc(&ctx->trt_in, info.GetNumBytes()));
            ck_cu(cudaMalloc(&ctx->frame_in, info.GetNumBytes() / sizeof(float)));
        }
        if (!info.bInput)
        {
            *output_height = info.dim.d[2];
            *output_width = info.dim.d[3];

            ck_cu(cudaMalloc(&ctx->trt_out, info.GetNumBytes()));
            ck_cu(cudaMalloc(&ctx->frame_out, info.GetNumBytes() / sizeof(float)));
        }
    }

    return DNN_SUCCESS;
}

DNNReturnType load_model_trt(DNNModel *model, TRTContext *ctx, const char *model_filename)
{
    cudaSetDevice(ctx->options.device);
    TrtLite *trt_model= new TrtLite{model_filename, ctx, logger};
    if (trt_model == nullptr)
    {
        return DNN_ERROR;
    }

    trt_model->PrintInfo();

    model->model = trt_model;
    model->get_input = &get_input_trt;
    model->get_output = &get_output_trt;

    return DNN_SUCCESS;
}

DNNReturnType execute_model_trt(const DNNModel *model, uint8_t **in_frame_data, int *in_linesize,
                                        int in_width, int in_height, uint8_t **out_frame_data, int *out_linesize,
                                        int out_width, int out_height, int packed, cudaStream_t stream)
{
    TrtLite* trt_model = reinterpret_cast<TrtLite*>(model->model);
    TRTContext *ctx = trt_model->ctx;

    vector<void*> buf_vec, device_buf_vec;
    int ret = 0;

    int input_height = in_height;
    int input_width = in_width;
    int input_channels = ctx->channels;
    vector<IOInfo> IO_info_vec;
    map<int, Dims> i2shape;

    if (!trt_model->GetEngine()->hasImplicitBatchDimension())
    {
        i2shape.insert(make_pair(0, Dims{4, {BATCH, input_channels, input_height, input_width}}));
        IO_info_vec = trt_model->ConfigIO(i2shape);
    }
    else
    {
        IO_info_vec = trt_model->ConfigIO(BATCH);
    }

    for (auto info : IO_info_vec)
    {

        if (info.bInput)
        {
            ret = frame_to_dnn(in_frame_data, in_linesize, in_width, in_height,
                                ctx, info.GetNumBytes() / sizeof(float), stream, packed);

            if (ret < 0)
                return DNN_ERROR;

            device_buf_vec.push_back((void*)ctx->trt_in);
            continue;
        }
        else
        {
            device_buf_vec.push_back((void*)ctx->trt_out);
        }
    }

    if (!trt_model->GetEngine()->hasImplicitBatchDimension())
    {
        trt_model->Execute(i2shape, device_buf_vec, stream);
    }
    else
    {
        trt_model->Execute(BATCH, device_buf_vec, stream);
    }

    for (uint32_t i = 0; i < IO_info_vec.size(); i++)
    {
        if (!IO_info_vec[i].bInput)
        {
            ret = dnn_to_frame(out_frame_data, out_linesize, out_width, out_height,
                                ctx, IO_info_vec[i].GetNumBytes() / sizeof(float), stream, packed);
        }
    }

    return DNN_SUCCESS;
}

DNNReturnType free_model_trt(DNNModel *model)
{
    TrtLite* trt_model = static_cast<TrtLite*>(model->model);
    TRTContext *ctx = trt_model->ctx;

    delete trt_model;

    ck_cu(cudaFree(ctx->trt_in));
    ck_cu(cudaFree(ctx->trt_out));
    ck_cu(cudaFree(ctx->frame_in));
    ck_cu(cudaFree(ctx->frame_out));

    return DNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
