
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvCaffeParser.h"
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
const std::string gSampleName = "TensorRT.resnet18_api";



//!
//! \brief The SampleMNISTAPIParams structure groups the additional parameters required by
//!         the SampleMNISTAPI sample.
//!
struct ResNet18Params : public ResNetCommon::ResNetParams
{
    int inputH;                  //!< The input height
    int inputW;                  //!< The input width
    int outputSize;              //!< The output size
    std::string weightsFile;     //!< The filename of the weights file
    std::string mnistMeansProto; //!< The proto file containing means
};

//! \brief  The SampleResNet class implements the ResNet18 Network
//!
//! \details It creates the network for ResNet Classifier using the API
//!
class ResNet18
{
    template <typename T>
    using SampleUniquePt = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    ResNet18(const ResNet18Params& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    ResNet18Params mParams; //!< The parameters for the sample.
    static const int INPUT_H = 224;
    static const int INPUT_W = 224;
    static const int INPUT_C = 3;
    const int mNumber = "Cat"; //!< The number to classify

    std::map<std::string, nvinfer1::Weights> mWeightMap; //!< The weight name to weight value map

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    IScaleLayer*  addBatchNorm2d(
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
    ILayer& input,
    std::string layer_name,
    float eps)


    ILayer* basicBlock(
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
    std::map<std::string,
    Itensor& input,
    int in_channels,
    int out_channels,
    int stride,
    std::string lname)

    //!
    //! \brief Uses the API to create the MNIST Network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Loads weights from weights file
    //!
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the MNIST network by using the API to create a model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleMNISTAPI::build()
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


/*()
"""
ILayer * SampleMNISTAPI::addDepthwiseBlock(
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
    ILayer * input,
    int out_channel,
    int map_channel,
    int stride,
    bool residual,
    int layer_num)
{

    short * arr_heap, * arr1_heap, * arr2_heap, * arr3_heap, * arr4_heap, *arr5_heap;


    
    nvinfer1::Weights conv_weight, bias_weight;
    string filename = "mbnetv2/expand_1x1_" + to_string(layer_num) + ".npy";
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    conv_weight.count = arr.shape[0] * arr.shape[1];
    string filename1 = mbnetv2/expand_1x1_ + to_string(layer_num) + "_bias.npy";
    cnpy::NpyArray arr1 = cnpy::npy_load(filename1);
    bias_weight.count = arr1.shape[0];

    arr_heap = new short[conv_weight.count];
    arr1_heap = new short[bias_weight.count];
    memcpy(arr_heap, arr.data<short>(),conv_weight.count * 2);
    memcpy(arr1_heap, arr1.data<short>(),bias_weight.count * 2);
    conv_weight.values = arr_heap;
    bias_weight.values = arr1_heap;
    conv_weight.type = DataType::kHALF;
    bias_weight.type = DataType::kHALF;

    IConvolutionLayer* expansion = network->addConvolution(*input->getOutput(0), map_channel, DimsHW{1, 1}, conv_weight, bias_weight);
    assert(expansion);
    expansion->setStride(DimsHW{1, 1});
    expansion->setPadding(DimsHW{0, 0});
    
    IActivationLayer * relu = network->addActivation(*expansion->getOutput(0), ActivationType::kRELU);
    assert(relu);
    
    nvinfer1::Weights conv_weight_1, bias_weight_1;
    filename = "mbnetv2/depthwise_nxn_" + to_string(layer_num) + ".npy" ;
    arr = cnpy::npy_load(filename);
    conv_weight_1.count = arr.shape[0] * arr.shape[1] * arr.shape[2];
    filename1 = "mbnetv2/depthwise_nxn_" + to_string(layer_num) + "_bias.npy";
    arr1 = cnpy::npy_load(filename1);
    bias_weight_1.count = arr1.shape[0];

    arr2_heap = new short[map_channel * 9];
    arr3_heap = new short[map_channel];
    memcpy(arr2_heap, arr.data<short>(),conv_weight_1.count * 2);
    memcpy(arr3_heap, arr1.data<short>(),bias_weight_1.count * 2);
    conv_weight_1.values = arr2_heap;
    bias_weight_1.values = arr3_heap;
    conv_weight_1.type = DataType::kHALF;
    bias_weight_1.type = DataType::kHALF;
    IConvolutionLayer* depthwise = network->addConvolution(*relu->getOutput(0), map_channel, DimsHW{3, 3}, conv_weight_1, bias_weight_1);
    assert(depthwise);
    depthwise->setStride(DimsHW{stride, stride});
    depthwise->setPadding(DimsHW{1 , 1});
    depthwise->setNbGroups(map_channel);
    
    IActivationLayer * relu1 = network->addActivation(*depthwise->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    
    nvinfer1::Weights conv_weight_2, bias_weight_2;
    filename = "mbnetv2/contraction_1x1_" + to_string(layer_num) + ".npy";
    arr = cnpy::npy_load(filename);
    conv_weight_2.count = arr.shape[0] * arr.shape[1];
    filename1 = "mbnetv2/contraction_1x1_" + to_string(layer_num) + "_bias.npy";
    arr1 = cnpy::npy_load(filename1);
    bias_weight_2.count = arr1.shape[0];
    conv_weight_2.type = DataType::kHALF;
    bias_weight_2.type = DataType::kHALF;

    arr4_heap = new short[map_channel * out_channel];
    arr5_heap = new short[out_channel];
    memcpy(arr4_heap, arr.data<short>(),conv_weight_2.count * 2);
    memcpy(arr5_heap, arr1.data<short>(),bias_weight_2.count * 2);
    conv_weight_2.values = arr4_heap;
    bias_weight_2.values = arr5_heap;
    
    IConvolutionLayer* contraction = network->addConvolution(*relu1->getOutput(0), out_channel, DimsHW{1, 1}, conv_weight_2, bias_weight_2);
    assert(contraction);
    contraction->setStride(DimsHW{1, 1});
    contraction->setPadding(DimsHW{0, 0});
    

    if(!residual)
    {
       return contraction;
    }
    else
    {
       IElementWiseLayer * output = network->addElementWise(
		       *contraction->getOutput(0),
		       *input->getOutput(0),
		       ElementWiseOperation::kSUM
		       );
       assert(output);
       return output;
    }
    }
*/

//!
//! \brief Uses the API to create the MNIST Network
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleMNISTAPI::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    // Create input tensor of shape { 1, 1, 28, 28 }
    ITensor* data = network->addInput(
        mParams.inputTensorNames[0].c_str(), DataType::kHALF, Dims3{3,mParams.inputH, mParams.inputW});
    assert(data);

    // Create scale layer with default power/shift and specified scale parameter.
    /*const float scaleParam = 0.0125f;
    const Weights power{DataType::kHALF, nullptr, 0};
    const Weights shift{DataType::kHALF, nullptr, 0};
    const Weights scale{DataType::kHALF, &scaleParam, 1};
    IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
    assert(scale_1);*/
    //short * arr_heap, *arr1_heap;
    nvinfer1::Weights initial_conv_weight, initial_conv_bias;
    cnpy::NpyArray arr = cnpy::npy_load("mbnetv2/initial_conv.npy");
    initial_conv_weight.values = arr.data<short>();
    initial_conv_weight.count = arr.shape[0] * arr.shape[1] * arr.shape[2] * arr.shape[3];
    cnpy::NpyArray arr1 = cnpy::npy_load("mbnetv2/initial_conv_bias.npy");
    initial_conv_bias.count = arr1.shape[0];
    initial_conv_bias.values = arr1.data<short>();
/*
    arr_heap = new short[initial_conv_weight.count];
    arr1_heap = new short[initial_conv_bias.count];
    memcpy(arr_heap, arr.data<short>(), initial_conv_weight.count * 2);
    memcpy(arr1_heap, arr1.data<short>(), initial_conv_bias.count * 2);
    initial_conv_weight.values = arr_heap;
    initial_conv_bias.values = arr1_heap; */
    initial_conv_weight.type = DataType::kHALF;
    initial_conv_bias.type = DataType::kHALF;

    // Add convolution layer with 20 outputs and a 5x5 filter.
    IConvolutionLayer* conv1 = network->addConvolution(*data, 16, DimsHW{3, 3}, initial_conv_weight, initial_conv_bias);
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{1, 1});
    
    IActivationLayer* relu = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu);
  
    ILayer* inter;
    ILayer* result;
    //out, map, stride, residual?, layer_num
    inter = addDepthwiseBlock(network, relu, 16, 16, 1, true, 0);
    result = addDepthwiseBlock(network, inter, 24, 128, 2, false, 1);
    inter = addDepthwiseBlock(network, result, 24, 192, 1, true, 2);
    inter = addDepthwiseBlock(network, inter, 32, 192, 2, false, 3);
    inter = addDepthwiseBlock(network, inter, 32, 256, 1, true, 4);
    inter = addDepthwiseBlock(network, inter, 32, 256, 1, true, 5);
    inter = addDepthwiseBlock(network, inter, 32, 256, 1, true, 6);
    inter = addDepthwiseBlock(network, inter, 64, 128, 2, false, 7);
    inter = addDepthwiseBlock(network, inter, 64, 256, 1, true, 8);
    inter = addDepthwiseBlock(network, inter, 64, 256, 1, true, 9);
    inter = addDepthwiseBlock(network, inter, 64, 256, 1, true, 10);
    inter = addDepthwiseBlock(network, inter, 64, 256, 1, true, 11);
    inter = addDepthwiseBlock(network, inter, 64, 256, 1, true, 12);
    inter = addDepthwiseBlock(network, inter, 96, 192, 1, false, 13);
    inter = addDepthwiseBlock(network, inter, 96, 288, 1, true, 14);
    inter = addDepthwiseBlock(network, inter, 96, 288, 1, true, 15);
    inter = addDepthwiseBlock(network, inter, 96, 288, 1, true, 16);
    inter = addDepthwiseBlock(network, inter, 96, 288, 1, true, 17);
    inter = addDepthwiseBlock(network, inter, 96, 288, 1, true, 18);
    inter = addDepthwiseBlock(network, inter, 160, 192, 2, false, 19);
    inter = addDepthwiseBlock(network, inter, 160, 320, 1, true, 20);
    inter = addDepthwiseBlock(network, inter, 160, 320, 1, true, 21);
    inter = addDepthwiseBlock(network, inter, 160, 320, 1, true, 22);
    inter = addDepthwiseBlock(network, inter, 160, 320, 1, true, 23);
    inter = addDepthwiseBlock(network, inter, 160, 320, 1, true, 24);
    inter = addDepthwiseBlock(network, inter, 320, 960, 1, false, 25);

    nvinfer1::Weights final_conv_weight, final_conv_bias;
    auto arr2 = cnpy::npy_load("mbnetv2/final_1x1_conv.npy");
    final_conv_weight.values = arr2.data<short>();
    final_conv_weight.count = arr2.shape[0] * arr2.shape[1];
    auto arr3 = cnpy::npy_load("mbnetv2/final_1x1_conv_bias.npy");
    final_conv_bias.values = arr3.data<short>();
    final_conv_bias.count = arr3.shape[0];
    final_conv_weight.type = DataType::kHALF;
    final_conv_bias.type = DataType::kHALF;

    IConvolutionLayer* last_conv = network->addConvolution(*inter->getOutput(0), 1280, DimsHW{1,1}, final_conv_weight, final_conv_bias);
    assert(last_conv);
    last_conv->setStride(DimsHW{1, 1});
    last_conv->setPadding(DimsHW{0, 0});
    
    IActivationLayer* relu1 = network->addActivation(*last_conv->getOutput(0), ActivationType::kRELU);

    std::cout << relu1->getOutput(0)->getDimensions() << std::endl;
    IPoolingLayer* avg_pool = network->addPooling(*relu1->getOutput(0),PoolingType::kAVERAGE,DimsHW{7,7});
    assert(avg_pool);
    std::cout << avg_pool->getOutput(0)->getDimensions() << std::endl;
    nvinfer1::Weights fc_weight, fc_bias;
    auto arr4 = cnpy::npy_load("mbnetv2/final_dense_kernel.npy");
    fc_weight.values = arr4.data<short>();
    fc_weight.count = arr4.shape[0] * arr4.shape[1];
    auto arr5 = cnpy::npy_load("mbnetv2/final_dense_bias.npy");
    fc_bias.values = arr5.data<short>();
    fc_bias.count = arr5.shape[0];
    fc_weight.type = DataType::kHALF;
    fc_bias.type = DataType::kHALF;
    IFullyConnectedLayer * fc = network->addFullyConnected(*avg_pool->getOutput(0), 1000, fc_weight, fc_bias);

    // Add softmax layer to determine the probability.
    ISoftMaxLayer* prob = network->addSoftMax(*fc->getOutput(0));
    assert(prob);
    prob->getOutput(0)->setName(mParams.outputTensorNames[0].c_str());
    network->markOutput(*prob->getOutput(0));

    //avg_pool->getOutput(0)->setName(mParams.outputTensorNames[0].c_str());
    //network->markOutput(*avg_pool->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(64_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 64.0f, 64.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleMNISTAPI::infer()
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

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    
    for(int i = 0; i < 20; i ++){
    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float totalTime;
    cudaEventElapsedTime(&totalTime,start,end);
    std::cout << "Took: " << totalTime / 20 << " ms" << std::endl;


    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleMNISTAPI::processInput(const samplesCommon::BufferManager& buffers)
{
    cnpy::NpyArray arr = cnpy::npy_load("mbnetv2/test_image.npy");
    short * values = arr.data<short>();
    std::cout << values[0] << " " << values[1] << std::endl;
    short * hostDataBuffer = static_cast<short*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < mParams.inputH * mParams.inputW * 3; i++)
    {
        hostDataBuffer[i] = short(values[i]);
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleMNISTAPI::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    short* prob = static_cast<short*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    cnpy::npy_save("tensorrt_output.npy",&prob[0],{1000},"w");
    std::cout << prob[0] << " " << prob[1] << std::endl;
    std::cout << "\nOutput:\n"
              << std::endl;
    short maxVal{0};
    int idx{0};
    for (int i = 0; i < mParams.outputSize; i++)
    {
        if (maxVal < prob[i])
        {
            maxVal = prob[i];
            idx = i;
        }
    }
    std::cout <<"Prediction: " << std::endl;

    return idx == mNumber && maxVal > 0.9f;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleMNISTAPI::teardown()
{
    // Release weights host memory
    for (auto& mem : mWeightMap)
    {
        auto weight = mem.second;
        if (weight.type == DataType::kFLOAT)
        {
            delete[] static_cast<const uint32_t*>(weight.values);
        }
        else
        {
            delete[] static_cast<const uint16_t*>(weight.values);
        }
    }

    return true;
}


//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleMNISTAPIParams initializeSampleParams(const samplesCommon::Args& args)
{
    SampleMNISTAPIParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.inputTensorNames.push_back("data");
    params.batchSize = 1;
    params.outputTensorNames.push_back("prob");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    std::cout << args.runInFp16 << std::endl;
    params.inputH = 224;
    params.inputW = 224;
    params.outputSize = 1000;
    params.mnistMeansProto = "mnist_mean.binaryproto";

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_mnist_api [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleMNISTAPI sample(initializeSampleParams(args));

    gLogInfo << "Building and running a GPU inference engine for MNIST API" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
