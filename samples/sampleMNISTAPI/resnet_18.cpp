
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

bool ResNet18::build()
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

ILayer * ResNet18::addBatchNorm2d(
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
    ILayer& input,
    std::string layer_name,
    float eps)
    {
    








    }