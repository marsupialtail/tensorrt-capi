#include "argParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cnpy.h>
#include <cstring>
#include <chrono>

using namespace std::chrono;
using namespace std;

const std::string gSampleName = "TensorRT.sample_ssd_resnet34"

struct SampleSSDParams : public samplesCommon::SampleParams
{
    int outputClsSize;
    int keepTopK;
    float visualThreshold;
    int nbCalBatches;
    std::string calibrationBatches;
}

class SampleSSD
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleSSD(const SampleSSDParams& params) : mParams(params), mEngine(nullptr)
    {   
    }
}
    bool build();
    bool infer();
    bool teardown();

private:
    SampleSSDParams mParams;

    nvinfer1::Dims mInputDims;
    
    std::vector<samplesCommon::PPM<3, 300, 300>> mPPMs;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

    Ilayer * add ResBlock(
    NetworkDefinition *network,
    ILayer * input,
    int in_channels,
    int out_channels,
    int stride, 
    string layer_name)

    bool processInput(const samplesCommon::BufferManager& buffers);
    
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

bool SampleSSD::build()
{

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    auto inputDims = network->getInput(0)->getDimensions();
    assert(inputDims.nbDims == 3);

    assert(network->getNbOutputs() == 1);
    auto outputDims = network->getOutput(0)->getDimensions();
    assert(outputDims.nbDims == 3);

    return true;
}

Ilayer * add ResBlock(INetworkDefinition *network,
    ILayer * input,
    int in_channels,
    int out_channels,
    int stride, 
    string layer_name) {
    

    short * arr_heap, * arr1_heap, * arr2_heap, * arr3_heap, * arr4_heap, *arr5_heap, *arr6_heap, *arr7_heap, *arr8_heap;
    
    // Layer 1: 1x1 Convolution
    nvinfer1::Weights conv_weight_0, conv_bias_0;
    string conv0_weight_name = "weights/" + to_string(layer_name) + "1.npy";
    cnpy::NpyArray arr_wt0 = cnpy::npy_load(conv0_weight_name);
    conv_weight_0.count = arr_wt0.shape[0] * arr_wt0.shape[1];
    string conv0_bias_name = "weights/" + to_string(layer_name) + "1_bias.npy";
    cnpy::NpyArray arr_bias0 = cnpy::npy_load(conv0_bias_name);
    conv_bias_0.count = arr_bias0.shape[0];

    arr_heap = new short[conv_weight_0.count];
    arr1_heap = new short[conv_bias_0.count];
    memcpy(arr_heap, arr_wt0.data<short>(),conv_weight_0.count * 2);
    memcpy(arr1_heap, arr_bias0.data<short>(),conv_bias_0.count * 2);
    conv_weight_0.values = arr_heap;
    conv_bias_0.values = arr1_heap;
    conv_weight_0.type = DataType::kHALF;
    conv_bias_0.type = DataType::kHALF;

    IConvolutionLayer* conv1 = network->addConvolution(input, out_channels, DimsHW{1, 1}, conv_weight_0, conv_bias_0);
    assert(conv1);

    // ReLU 1
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    
    // Layer 2: 3x3 Convolution
    nvinfer1::Weights conv_weight_1, conv_bias_1;
    string conv1_weight_name = "weights/" + to_string(layer_name) + "2.npy";
    cnpy::NpyArray arr_wt1 = cnpy::npy_load(conv1_weight_name);
    conv_weight_1.count = arr_wt1.shape[0] * arr_wt1.shape[1];
    string conv1_bias_name = "weights/" + to_string(layer_name) + "2_bias.npy";
    cnpy::NpyArray arr_bias1 = cnpy::npy_load(conv1_bias_name);
    conv_bias_1.count = arr_bias1.shape[0];

    arr2_heap = new short[conv_weight_1.count];
    arr3_heap = new short[conv_bias_1.count];
    memcpy(arr2_heap, arr_wt1.data<short>(), conv_weight_1.count * 2);
    memcpy(arr3_heap, arr_bias1.data<short>(), conv_bias_1.count * 2);
    conv_weight_1.values = arr2_heap;
    bias_weight_1.values = arr3_heap;
    conv_weight_1.type = DataType::kHALF;
    bias_weight_1.type = DataType::kHALF;
    
    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), out_channels * 4, dimsHW{3, 3}, conv_weight_1, conv_bias_1); 
    assert(conv2);
    conv2->setStride(DimsHW{stride, stride});
    conv2->setPadding(DimsHW{1, 1});
    
    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationTyoe::kRELU);
    assert(relu2);

    //Layer 3: 1x1 Convolution

    nvinfer1::Weights conv_weight_2, conv_bias_2; 
    string conv2_weight_name = "weights/" + to_string(layer_name) + "3.npy";
    cnpy::NpyArray arr_wt2 = cnpy::npy_load(conv2_weight_name);
    conv_weight_2.count = arr_wt2.shape[0] * arr_wt2.shape[1];
    string conv2_bias_name = "weights/" + to_string(layer_name) + "3_bias.npy";
    cnpy::NpyArray arr_bias1 = cnpy::npy_load(conv2_bias_name);
    conv_bias_2.count = arr_bias2.shape[0];

    arr2_heap = new short[conv_weight_2.count];
    arr3_heap = new short[conv_bias_2.count];
    memcpy(arr2_heap, arr_wt2.data<short>(), conv_weight_1.count * 2);
    memcpy(arr3_heap, arr1_bias2.data<short>(), conv_bias_1.count * 2);
    conv_weight_1.values = arr2_heap;
    bias_weight_1.values = arr3_heap;
    conv_weight_1.type = DataType::kHALF;
    bias_weight_1.type = DataType::kHALF;
    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), out_channels * 4, DimsHW{1, 1}, conv_weight_2, conv_bias_2);
    assert(conv3);

    // If Downsample
    IElementWiseLayer* ew1;
    if (stride != 1 || in_channels != out_channels * 4) {
        nvinfer1::Weights downsample_weight_0, downsample_bias_0;
        IConvolutionLayer* conv4 = network->addConvolution(input, out_channels * 4, Dims{1, 1}, downsample_weight_0, downsample_bias_0);
        assert(conv4);
        conv4->setStride(DimsHW{stride, stride});
        ew1 = network->addElementWise(*conv4->getOutput(0), *conv3->getOutput(0), ElementWiseOperation::kSUM);
    }
    else {
        ew1 = network->addElementWise(input, *conv3->getOutput(0), ElementWiseOperation::kSum);
    }

    // ReLU 3
    IActivationLayer* relu3 = network->addACtivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
    
    }

 







bool SampleSSD::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}