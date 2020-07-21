/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorNet.h"
#include "randInt8Calibrator.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"
#include "filesystem.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#if NV_TENSORRT_MAJOR >= 5
#include "NvOnnxParser.h"
#include "NvUffParser.h"
// #include "NvInferPlugin.h"
#endif

#include <fstream>
#include <iostream>
#include <map>

#include <cassert>
#include <cublas_v2.h>
#include <cudnn.h>
#include <sstream>
#include <bits/stl_algo.h>
#include <bits/stl_numeric.h>

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)



#if NV_TENSORRT_MAJOR > 1
#  define CREATE_INFER_BUILDER nvinfer1::createInferBuilder
#  define CREATE_INFER_RUNTIME nvinfer1::createInferRuntime
#else
#  define CREATE_INFER_BUILDER createInferBuilder
#  define CREATE_INFER_RUNTIME createInferRuntime
#endif

#define LOG_DOWNLOADER_TOOL "        if loading a built-in model, maybe it wasn't downloaded before.\n\n"    \
					   "        Run the Model Downloader tool again and select it for download:\n\n"   \
					   "           $ cd <jetson-inference>/tools\n" 	  	\
					   "           $ ./download-models.sh\n"

//---------------------------------------------------------------------


const char* precisionTypeToStr(precisionType type) {
  switch (type) {
    case TYPE_DISABLED:
      return "DISABLED";
    case TYPE_FASTEST:
      return "FASTEST";
    case TYPE_FP32:
      return "FP32";
    case TYPE_FP16:
      return "FP16";
    case TYPE_INT8:
      return "INT8";
  }
}

precisionType precisionTypeFromStr(const char* str) {
  if (!str) return TYPE_DISABLED;

  for (int n = 0; n < NUM_PRECISIONS; n++) {
    if (strcasecmp(str, precisionTypeToStr((precisionType)n)) == 0)
      return (precisionType)n;
  }

  return TYPE_DISABLED;
}

static inline nvinfer1::DataType precisionTypeToTRT(precisionType type) {
  switch (type) {
    case TYPE_FP16:
      return nvinfer1::DataType::kHALF;
#if NV_TENSORRT_MAJOR >= 4
    case TYPE_INT8:
      return nvinfer1::DataType::kINT8;
#endif
  }

  return nvinfer1::DataType::kFLOAT;
}

static inline bool isFp16Enabled(nvinfer1::IBuilder* builder) {
#if NV_TENSORRT_MAJOR < 4
  return builder->getHalf2Mode();
#else
  return builder->getFp16Mode();
#endif
}

static inline bool isInt8Enabled(nvinfer1::IBuilder* builder) {
#if NV_TENSORRT_MAJOR >= 4
  return builder->getInt8Mode();
#else
  return false;
#endif
}

#if NV_TENSORRT_MAJOR >= 4
static inline const char* dataTypeToStr(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return "FP32";
    case nvinfer1::DataType::kHALF:
      return "FP16";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
  }
}

static inline const char* dimensionTypeToStr(nvinfer1::DimensionType type) {
  switch (type) {
    case nvinfer1::DimensionType::kSPATIAL:
      return "SPATIAL";
    case nvinfer1::DimensionType::kCHANNEL:
      return "CHANNEL";
    case nvinfer1::DimensionType::kINDEX:
      return "INDEX";
    case nvinfer1::DimensionType::kSEQUENCE:
      return "SEQUENCE";
  }
}
#endif

#if NV_TENSORRT_MAJOR > 1
static inline nvinfer1::Dims validateDims(const nvinfer1::Dims& dims) {
  if (dims.nbDims == nvinfer1::Dims::MAX_DIMS) return dims;

  nvinfer1::Dims dims_out = dims;

  // TRT doesn't set the higher dims, so make sure they are 1
  for (int n = dims_out.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++)
    dims_out.d[n] = 1;

  return dims_out;
}
#endif

const char* deviceTypeToStr(deviceType type) {
  switch (type) {
    case DEVICE_GPU:
      return "GPU";
    case DEVICE_DLA_0:
      return "DLA_0";
    case DEVICE_DLA_1:
      return "DLA_1";
  }
}

deviceType deviceTypeFromStr(const char* str) {
  if (!str) return DEVICE_GPU;

  for (int n = 0; n < NUM_DEVICES; n++) {
    if (strcasecmp(str, deviceTypeToStr((deviceType)n)) == 0)
      return (deviceType)n;
  }

  if (strcasecmp(str, "DLA") == 0) return DEVICE_DLA;

  return DEVICE_GPU;
}

#if NV_TENSORRT_MAJOR >= 5
static inline nvinfer1::DeviceType deviceTypeToTRT(deviceType type) {
  switch (type) {
    case DEVICE_GPU:
      return nvinfer1::DeviceType::kGPU;
      //case DEVICE_DLA:	return nvinfer1::DeviceType::kDLA;
#  if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0
    case DEVICE_DLA_0:
      return nvinfer1::DeviceType::kDLA0;
    case DEVICE_DLA_1:
      return nvinfer1::DeviceType::kDLA1;
#  else
    case DEVICE_DLA_0:
      return nvinfer1::DeviceType::kDLA;
    case DEVICE_DLA_1:
      return nvinfer1::DeviceType::kDLA;
#  endif
  }
}
#endif

const char* modelFormatToStr(modelFormat format) {
  switch (format) {
    case MODEL_CUSTOM:
      return "custom";
    case MODEL_CAFFE:
      return "caffe";
    case MODEL_ONNX:
      return "ONNX";
    case MODEL_UFF:
      return "UFF";
  }
}

modelFormat modelFormatFromStr(const char* str) {
  if (!str) return MODEL_CUSTOM;

  if (strcasecmp(str, "caffemodel") == 0 || strcasecmp(str, "caffe") == 0)
    return MODEL_CAFFE;
  else if (strcasecmp(str, "onnx") == 0)
    return MODEL_ONNX;
  else if (strcasecmp(str, "uff") == 0)
    return MODEL_UFF;

  return MODEL_CUSTOM;
}
//---------------------------------------------------------------------

// constructor
tensorNet::tensorNet() {
  mEngine = NULL;
  mInfer = NULL;
  mContext = NULL;
  mStream = NULL;

  mWidth = 0;
  mHeight = 0;
  mInputSize = 0;
  mMaxBatchSize = 0;
  mInputCPU = NULL;
  mInputCUDA = NULL;
  mEnableDebug = false;
  mEnableProfiler = false;

  mModelFormat = MODEL_CUSTOM;
  mPrecision = TYPE_FASTEST;
  mDevice = DEVICE_GPU;
  mAllowGPUFallback = false;

  memset(mEvents, 0, sizeof(mEvents));

#if NV_TENSORRT_MAJOR < 2
  memset(&mInputDims, 0, sizeof(Dims3));
#endif
}

// Destructor
tensorNet::~tensorNet() {
  if (mEngine != NULL) {
    mEngine->destroy();
    mEngine = NULL;
  }

  if (mInfer != NULL) {
    mInfer->destroy();
    mInfer = NULL;
  }
}

// EnableProfiler
void tensorNet::EnableProfiler() {
  mEnableProfiler = true;

  if (mContext != NULL) mContext->setProfiler(&gProfiler);
}

// EnableDebug
void tensorNet::EnableDebug() { mEnableDebug = true; }

// DetectNativePrecisions()
std::vector<precisionType> tensorNet::DetectNativePrecisions(
    deviceType device) {
  std::vector<precisionType> types;
  Logger logger;

  // create a temporary builder for querying the supported types
  nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(logger);

  if (!builder) {
    printf(LOG_TRT
           "QueryNativePrecisions() failed to create TensorRT IBuilder "
           "instance\n");
    return types;
  }

#if NV_TENSORRT_MAJOR >= 5
  if (device == DEVICE_DLA_0 || device == DEVICE_DLA_1)
    builder->setFp16Mode(true);

  builder->setDefaultDeviceType(deviceTypeToTRT(device));
#endif

  // FP32 is supported on all platforms
  types.push_back(TYPE_FP32);

  // detect fast (native) FP16
  if (builder->platformHasFastFp16()) types.push_back(TYPE_FP16);

#if NV_TENSORRT_MAJOR >= 4
  // detect fast (native) INT8
  if (builder->platformHasFastInt8()) types.push_back(TYPE_INT8);
#endif

  // print out supported precisions (optional)
  const uint32_t numTypes = types.size();

  printf(LOG_TRT "native precisions detected for %s:  ",
         deviceTypeToStr(device));

  for (uint32_t n = 0; n < numTypes; n++) {
    printf("%s", precisionTypeToStr(types[n]));

    if (n < numTypes - 1) printf(", ");
  }

  printf("\n");
  builder->destroy();
  return types;
}

// DetectNativePrecision
bool tensorNet::DetectNativePrecision(const std::vector<precisionType>& types,
                                      precisionType type) {
  const uint32_t numTypes = types.size();

  for (uint32_t n = 0; n < numTypes; n++) {
    if (types[n] == type) return true;
  }

  return false;
}

// DetectNativePrecision
bool tensorNet::DetectNativePrecision(precisionType precision,
                                      deviceType device) {
  std::vector<precisionType> types = DetectNativePrecisions(device);
  return DetectNativePrecision(types, precision);
}

// Create an optimized GIE network from caffe prototxt and model file
bool tensorNet::ProfileModel(const std::string& deployFile,			   // name for caffe prototxt
					    const std::string& modelFile,			   // name for model 
					    const char* input, const Dims3& inputDims,
					    const std::vector<std::string>& outputs,    // network outputs
					    unsigned int maxBatchSize,			   // batch size - NB must be at least as large as the batch we want to run with
					    precisionType precision, 
					    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, 	
					    std::ostream& gieModelStream)			   // output stream for the GIE model
{
	// create API root class - must span the lifetime of the engine usage
	nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	builder->setDebugSync(mEnableDebug);
	builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
	builder->setAverageFindIterations(2);

	//mEnableFP16 = (mOverride16 == true) ? false : builder->platformHasFastFp16();
	//printf(LOG_TRT "platform %s fast FP16 support\n", mEnableFP16 ? "has" : "does not have");
	printf(LOG_TRT "device %s, loading %s %s\n", deviceTypeToStr(device), deployFile.c_str(), modelFile.c_str());
	

	// parse the different types of model formats
	if( mModelFormat == MODEL_CAFFE )
	{
		// parse the caffe model to populate the network, then set the outputs
		nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

		nvinfer1::DataType modelDataType = (precision == TYPE_FP16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // import INT8 weights as FP32
		const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor =
			parser->parse(deployFile.c_str(),		// caffe deploy file
						  modelFile.c_str(),	// caffe model file
						 *network,			// network definition that the parser will populate
						  modelDataType);

		if( !blobNameToTensor )
		{
			printf(LOG_TRT "device %s, failed to parse caffe network\n", deviceTypeToStr(device));
			return false;
		}

		// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
		const size_t num_outputs = outputs.size();
		
		for( size_t n=0; n < num_outputs; n++ )
		{
			nvinfer1::ITensor* tensor = blobNameToTensor->find(outputs[n].c_str());
		
			if( !tensor )
				printf(LOG_TRT "failed to retrieve tensor for Output \"%s\"\n", outputs[n].c_str());
			else
			{
			#if NV_TENSORRT_MAJOR >= 4
				nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(tensor->getDimensions());
				printf(LOG_TRT "retrieved Output tensor \"%s\":  %ix%ix%i\n", tensor->getName(), dims.d[0], dims.d[1], dims.d[2]);
			#endif
			}

			network->markOutput(*tensor);
		}

		//parser->destroy();
	}
#if NV_TENSORRT_MAJOR >= 5
	else if( mModelFormat == MODEL_ONNX )
	{
    #if NV_TENSORRT_MAJOR >= 7
        network->destroy();
        network = builder->createNetworkV2(1U << (uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        if( !network )
        {
            printf(LOG_TRT "IBuilder::createNetworkV2(EXPLICIT_BATCH) failed\n");
            return false;
        }
    #endif

		nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

		if( !parser )
		{
			printf(LOG_TRT "failed to create nvonnxparser::IParser instance\n");
			return false;
		}

    #if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0
        const int parserLogLevel = (int)nvinfer1::ILogger::Severity::kINFO;
    #else
        const int parserLogLevel = (int)nvinfer1::ILogger::Severity::kVERBOSE;
    #endif

		if( !parser->parseFromFile(modelFile.c_str(), parserLogLevel) )
		{
			printf(LOG_TRT "failed to parse ONNX model '%s'\n", modelFile.c_str());
			return false;
		}

		//parser->destroy();
	}
	else if( mModelFormat == MODEL_UFF )
	{
		// create parser instance
		nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
		
		if( !parser )
		{
			printf(LOG_TRT "failed to create UFF parser\n");
			return false;
		}
		
		// register input
		if( !parser->registerInput(input, inputDims, nvuffparser::UffInputOrder::kNCHW) )
		{
			printf(LOG_TRT "failed to register input '%s' for UFF model '%s'\n", input, modelFile.c_str());
			return false;
		}
		
		// register outputs
		/*const size_t numOutputs = outputs.size();
		
		for( uint32_t n=0; n < numOutputs; n++ )
		{
			if( !parser->registerOutput(outputs[n].c_str()) )
				printf(LOG_TRT "failed to register output '%s' for UFF model '%s'\n", outputs[n].c_str(), modelFile.c_str());
		}*/

		if( !parser->registerOutput("MarkOutput_0") )
			printf(LOG_TRT "failed to register output '%s' for UFF model '%s'\n", "MarkOutput_0", modelFile.c_str());

		
		// parse network
		if( !parser->parse(modelFile.c_str(), *network, nvinfer1::DataType::kFLOAT) )
		{
			printf(LOG_TRT "failed to parse UFF model '%s'\n", modelFile.c_str());
			return false;
		}
		
		//parser->destroy();
	}
#endif

	// build the engine
	printf(LOG_TRT "device %s, configuring CUDA engine\n", deviceTypeToStr(device));
		
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);


	// set up the builder for the desired precision
	if( precision == TYPE_INT8 )
	{
	#if NV_TENSORRT_MAJOR >= 4
		builder->setInt8Mode(true);
		//builder->setFp16Mode(true);		// TODO:  experiment for benefits of both INT8/FP16
		
		if( !calibrator )
		{
	        // extract the dimensions of the network input blobs
	        std::map<std::string, nvinfer1::Dims3> inputDimensions;

	        for( int i=0, n=network->getNbInputs(); i < n; i++ )
	        {
                nvinfer1::Dims dims = network->getInput(i)->getDimensions();

            #if NV_TENSORRT_MAJOR >= 7
                if( mModelType == MODEL_ONNX )
                    dims = shiftDims(dims);  // change NCHW to CHW for EXPLICIT_BATCH
            #endif

		        //nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
		        inputDimensions.insert(std::make_pair(network->getInput(i)->getName(), static_cast<nvinfer1::Dims3&&>(dims)));
		        std::cout << LOG_TRT << "retrieved Input tensor \"" << network->getInput(i)->getName() << "\":  " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
	        }

            // default to random calibration
			calibrator = new randInt8Calibrator(1, mCacheCalibrationPath, inputDimensions);
			printf(LOG_TRT "warning:  device %s using INT8 precision with RANDOM calibration\n", deviceTypeToStr(device));
		}

		builder->setInt8Calibrator(calibrator);
	#else
		printf(LOG_TRT "INT8 precision requested, and TensorRT %u.%u doesn't meet minimum version for INT8\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		printf(LOG_TRT "please use minumum version of TensorRT 4.0 or newer for INT8 support\n");

		return false;
	#endif
	}
	else if( precision == TYPE_FP16 )
	{
	#if NV_TENSORRT_MAJOR < 4
		builder->setHalf2Mode(true);
	#else
		builder->setFp16Mode(true);
	#endif
	}
	

	// set the default device type
#if NV_TENSORRT_MAJOR >= 5
	builder->setDefaultDeviceType(deviceTypeToTRT(device));

	if( allowGPUFallback )
		builder->allowGPUFallback(true);
	
#if !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0)
	if( device == DEVICE_DLA_0 )
		builder->setDLACore(0);
	else if( device == DEVICE_DLA_1 )
		builder->setDLACore(1);
#endif
#else
	if( device != DEVICE_GPU )
	{
		printf(LOG_TRT "device %s is not supported in TensorRT %u.%u\n", deviceTypeToStr(device), NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
#endif

	// build CUDA engine
	printf(LOG_TRT "device %s, building FP16:  %s\n", deviceTypeToStr(device), isFp16Enabled(builder) ? "ON" : "OFF"); 
	printf(LOG_TRT "device %s, building INT8:  %s\n", deviceTypeToStr(device), isInt8Enabled(builder) ? "ON" : "OFF"); 
	printf(LOG_TRT "device %s, building CUDA engine (this may take a few minutes the first time a network is loaded)\n", deviceTypeToStr(device));

	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	
	if( !engine )
	{
		printf(LOG_TRT "device %s, failed to build CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	printf(LOG_TRT "device %s, completed building CUDA engine\n", deviceTypeToStr(device));

	// we don't need the network definition any more, and we can destroy the parser
	network->destroy();
	//parser->destroy();

	// serialize the engine, then close everything down
#if NV_TENSORRT_MAJOR > 1
	nvinfer1::IHostMemory* serMem = engine->serialize();

	if( !serMem )
	{
		printf(LOG_TRT "device %s, failed to serialize CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	gieModelStream.write((const char*)serMem->data(), serMem->size());
#else
	engine->serialize(gieModelStream);
#endif

	engine->destroy();
	builder->destroy();
	return true;
}


// FindFastestPrecision
precisionType tensorNet::FindFastestPrecision(deviceType device,
                                              bool allowInt8) {
  std::vector<precisionType> types = DetectNativePrecisions(device);

  if (allowInt8 && DetectNativePrecision(types, TYPE_INT8))
    return TYPE_INT8;
  else if (DetectNativePrecision(types, TYPE_FP16))
    return TYPE_FP16;
  else
    return TYPE_FP32;
}

// Create an optimized GIE network from caffe prototxt and model file
bool tensorNet::ProfileModel(
    const std::string& deployFile, // name for caffe prototxt
    const std::string& modelFile, // name for model
    const std::vector<std::string>& outputs, // network outputs
    unsigned int
        maxBatchSize, // batch size - NB must be at least as large as the batch we want to run with
    precisionType precision, deviceType device, bool allowGPUFallback,
    nvinfer1::IInt8Calibrator* calibrator,
    std::ostream& gieModelStream) // output stream for the GIE model
{
  // create API root class - must span the lifetime of the engine usage
  nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetwork();

  builder->setDebugSync(mEnableDebug);
  builder->setMinFindIterations(3); // allow time for TX1 GPU to spin up
  builder->setAverageFindIterations(2);

  //mEnableFP16 = (mOverride16 == true) ? false : builder->platformHasFastFp16();
  //printf(LOG_GIE "platform %s fast FP16 support\n", mEnableFP16 ? "has" : "does not have");
  printf(LOG_GIE "device %s, loading %s %s\n", deviceTypeToStr(device),
         deployFile.c_str(), modelFile.c_str());

  // parse the different types of model formats
  if (mModelFormat == MODEL_CAFFE) {
    // parse the caffe model to populate the network, then set the outputs
    nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

    nvinfer1::DataType modelDataType =
        (precision == TYPE_FP16)
            ? nvinfer1::DataType::kHALF
            : nvinfer1::DataType::kFLOAT; // import INT8 weights as FP32
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
        deployFile.c_str(), // caffe deploy file
        modelFile.c_str(), // caffe model file
        *network, // network definition that the parser will populate
        modelDataType);

    if (!blobNameToTensor) {
      printf(LOG_GIE "device %s, failed to parse caffe network\n",
             deviceTypeToStr(device));
      return false;
    }

    // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate
    const size_t num_outputs = outputs.size();

    for (size_t n = 0; n < num_outputs; n++) {
      nvinfer1::ITensor* tensor = blobNameToTensor->find(outputs[n].c_str());

      if (!tensor)
        printf(LOG_GIE "failed to retrieve tensor for Output \"%s\"\n",
               outputs[n].c_str());
      else {
#if NV_TENSORRT_MAJOR >= 4
        nvinfer1::Dims3 dims =
            static_cast<nvinfer1::Dims3&&>(tensor->getDimensions());
        printf(LOG_GIE "retrieved Output tensor \"%s\":  %ix%ix%i\n",
               tensor->getName(), dims.d[0], dims.d[1], dims.d[2]);
#endif
      }

      network->markOutput(*tensor);
    }

    //parser->destroy();
  }
#if NV_TENSORRT_MAJOR >= 5
  else if (mModelFormat == MODEL_ONNX) {
    nvonnxparser::IParser* parser =
        nvonnxparser::createParser(*network, gLogger);

    if (!parser) {
      printf(LOG_TRT "failed to create nvonnxparser::IParser instance\n");
      return false;
    }

    if (!parser->parseFromFile(modelFile.c_str(),
                               (int)nvinfer1::ILogger::Severity::kWARNING)) {
      printf(LOG_TRT "failed to parse ONNX model '%s'\n", modelFile.c_str());
      return false;
    }

    //parser->destroy();
  }
#endif

#if NV_TENSORRT_MAJOR >= 4
  // extract the dimensions of the network input blobs
  std::map<std::string, nvinfer1::Dims3> inputDimensions;

  for (int i = 0, n = network->getNbInputs(); i < n; i++) {
    nvinfer1::Dims3 dims =
        static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
    inputDimensions.insert(
        std::make_pair(network->getInput(i)->getName(), dims));
    std::cout << LOG_TRT << "retrieved Input tensor \""
              << network->getInput(i)->getName() << "\":  " << dims.d[0] << "x"
              << dims.d[1] << "x" << dims.d[2] << std::endl;
  }
#endif

  // build the engine
  printf(LOG_GIE "device %s, configuring CUDA engine\n",
         deviceTypeToStr(device));

  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(16 << 20);

  // set up the builder for the desired precision
  if (precision == TYPE_INT8) {
#if NV_TENSORRT_MAJOR >= 4
    builder->setInt8Mode(true);
    //builder->setFp16Mode(true);		// TODO:  experiment for benefits of both INT8/FP16

    if (!calibrator) {
      calibrator =
          new randInt8Calibrator(1, mCacheCalibrationPath, inputDimensions);
      printf(
          LOG_TRT
          "warning:  device %s using INT8 precision with RANDOM calibration\n",
          deviceTypeToStr(device));
    }

    builder->setInt8Calibrator(calibrator);
#else
    printf(LOG_TRT
           "INT8 precision requested, and TensorRT %u.%u doesn't meet minimum "
           "version for INT8\n",
           NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
    printf(LOG_TRT
           "please use minumum version of TensorRT 4.0 or newer for INT8 "
           "support\n");

    return false;
#endif
  } else if (precision == TYPE_FP16) {
#if NV_TENSORRT_MAJOR < 4
    builder->setHalf2Mode(true);
#else
    builder->setFp16Mode(true);
#endif
  }

  // set the default device type
#if NV_TENSORRT_MAJOR >= 5
  builder->setDefaultDeviceType(deviceTypeToTRT(device));

  if (allowGPUFallback) builder->allowGPUFallback(true);

#  if !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && \
        NV_TENSORRT_PATCH == 0)
  if (device == DEVICE_DLA_0)
    builder->setDLACore(0);
  else if (device == DEVICE_DLA_1)
    builder->setDLACore(1);
#  endif
#else
  if (device != DEVICE_GPU) {
    printf(LOG_TRT "device %s is not supported in TensorRT %u.%u\n",
           deviceTypeToStr(device), NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
    return false;
  }
#endif

  // build CUDA engine
  printf(LOG_TRT "device %s, building FP16:  %s\n", deviceTypeToStr(device),
         isFp16Enabled(builder) ? "ON" : "OFF");
  printf(LOG_TRT "device %s, building INT8:  %s\n", deviceTypeToStr(device),
         isInt8Enabled(builder) ? "ON" : "OFF");
  printf(LOG_GIE
         "device %s, building CUDA engine (this may take a few minutes the "
         "first time a network is loaded)\n",
         deviceTypeToStr(device));

  nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

  if (!engine) {
    printf(LOG_GIE "device %s, failed to build CUDA engine\n",
           deviceTypeToStr(device));
    return false;
  }

  printf(LOG_GIE "device %s, completed building CUDA engine\n",
         deviceTypeToStr(device));

  // we don't need the network definition any more, and we can destroy the parser
  network->destroy();
  //parser->destroy();

  // serialize the engine, then close everything down
#if NV_TENSORRT_MAJOR > 1
  nvinfer1::IHostMemory* serMem = engine->serialize();

  if (!serMem) {
    printf(LOG_GIE "device %s, failed to serialize CUDA engine\n",
           deviceTypeToStr(device));
    return false;
  }

  gieModelStream.write((const char*)serMem->data(), serMem->size());
#else
  engine->serialize(gieModelStream);
#endif

  engine->destroy();
  builder->destroy();
  return true;
}

// LoadNetwork
bool tensorNet::LoadNetwork(const char* prototxt_path, const char* model_path,
                            const char* mean_path, const char* input_blob,
                            const char* output_blob, uint32_t maxBatchSize,
                            precisionType precision, deviceType device,
                            bool allowGPUFallback,
                            nvinfer1::IInt8Calibrator* calibrator,
                            cudaStream_t stream) {
  std::vector<std::string> outputs;
  outputs.push_back(output_blob);

  return LoadNetwork(prototxt_path, model_path, mean_path, input_blob, outputs,
                     maxBatchSize, precision, device, allowGPUFallback);
}

// LoadNetwork
bool tensorNet::LoadNetwork(const char* prototxt_path_, const char* model_path_,
                            const char* mean_path, const char* input_blob,
                            const std::vector<std::string>& output_blobs,
                            uint32_t maxBatchSize, precisionType precision,
                            deviceType device, bool allowGPUFallback,
                            nvinfer1::IInt8Calibrator* calibrator,
                            cudaStream_t stream) {
  if (/*!prototxt_path_ ||*/ !model_path_) return false;

#if NV_TENSORRT_MAJOR >= 4
  printf(LOG_GIE "TensorRT version %u.%u.%u\n", NV_TENSORRT_MAJOR,
         NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
#else
  printf(LOG_GIE "TensorRT version %u.%u\n", NV_TENSORRT_MAJOR,
         NV_TENSORRT_MINOR);
#endif
 

  /*
	 * verify the prototxt and model paths
	 */
  const std::string model_path = locateFile(model_path_);
  const std::string prototxt_path =
      locateFile(prototxt_path_ != NULL ? prototxt_path_ : "");

  const std::string model_ext = fileExtension(model_path_);
  const modelFormat model_fmt = modelFormatFromStr(model_ext.c_str());

  printf(LOG_TRT "detected model format - %s  (extension '.%s')\n",
         modelFormatToStr(model_fmt), model_ext.c_str());

  if (model_fmt == MODEL_CUSTOM || model_fmt == MODEL_UFF) {
    printf(LOG_TRT "model format '%s' not supported by jetson-inference\n",
           modelFormatToStr(model_fmt));
    return false;
  }
#if NV_TENSORRT_MAJOR < 5
  else if (model_fmt == MODEL_ONNX) {
    printf(LOG_TRT
           "importing ONNX models is not supported in TensorRT %u.%u (version "
           ">= 5.0 required)\n",
           NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
    return false;
  }
#endif
  else if (model_fmt == MODEL_CAFFE && !prototxt_path_) {
    printf(LOG_TRT
           "attempted to load caffe model without specifying prototxt file\n");
    return false;
  }

  mModelFormat = model_fmt;

  /*
	 * if the precision is left unspecified, detect the fastest
	 */
  printf(LOG_TRT "desired precision specified for %s: %s\n",
         deviceTypeToStr(device), precisionTypeToStr(precision));

  if (precision == TYPE_DISABLED) {
    printf(LOG_TRT "skipping network specified with precision TYPE_DISABLE\n");
    printf(LOG_TRT "please specify a valid precision to create the network\n");

    return false;
  } else if (precision == TYPE_FASTEST) {
    if (!calibrator)
      printf(LOG_TRT
             "requested fasted precision for device %s without providing valid "
             "calibrator, disabling INT8\n",
             deviceTypeToStr(device));

    precision = FindFastestPrecision(device, (calibrator != NULL));
    printf(LOG_TRT "selecting fastest native precision for %s:  %s\n",
           deviceTypeToStr(device), precisionTypeToStr(precision));
  } else {
    if (!DetectNativePrecision(precision, device)) {
      printf(LOG_TRT "precision %s is not supported for device %s\n",
             precisionTypeToStr(precision), deviceTypeToStr(device));
      return false;
    }

    if (precision == TYPE_INT8 && !calibrator)
      printf(
          LOG_TRT
          "warning:  device %s using INT8 precision with RANDOM calibration\n",
          deviceTypeToStr(device));
  }

  /*
	 * attempt to load network from cache before profiling with tensorRT
	 */
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);

  char cache_prefix[512];
  char cache_path[512];

  sprintf(cache_prefix, "%s.%u.%u.%s.%s", model_path.c_str(), maxBatchSize,
          (uint32_t)allowGPUFallback, deviceTypeToStr(device),
          precisionTypeToStr(precision));
  sprintf(cache_path, "%s.calibration", cache_prefix);
  mCacheCalibrationPath = cache_path;

  sprintf(cache_path, "%s.engine", cache_prefix);
  mCacheEnginePath = cache_path;
  printf(LOG_GIE "attempting to open engine cache file %s\n",
         mCacheEnginePath.c_str());

  std::ifstream cache(mCacheEnginePath);

  if (!cache) {
    printf(LOG_GIE
           "cache file not found, profiling network model on device %s\n",
           deviceTypeToStr(device));

    if (!ProfileModel(prototxt_path, model_path, output_blobs, maxBatchSize,
                      precision, device, allowGPUFallback, calibrator,
                      gieModelStream)) {
      printf("device %s, failed to load %s\n", deviceTypeToStr(device),
             model_path.c_str());
      return 0;
    }

    printf(LOG_GIE "network profiling complete, writing engine cache to %s\n",
           mCacheEnginePath.c_str());
    std::ofstream outFile;
    outFile.open(mCacheEnginePath);
    outFile << gieModelStream.rdbuf();
    outFile.close();
    gieModelStream.seekg(0, gieModelStream.beg);
    printf(LOG_GIE "device %s, completed writing engine cache to %s\n",
           deviceTypeToStr(device), mCacheEnginePath.c_str());
  } else {
    printf(LOG_GIE "loading network profile from engine cache... %s\n",
           mCacheEnginePath.c_str());
    gieModelStream << cache.rdbuf();
    cache.close();

    // test for half FP16 support
    /*nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
		
		if( builder != NULL )
		{
			mEnableFP16 = !mOverride16 && builder->platformHasFastFp16();
			printf(LOG_GIE "platform %s fast FP16 support\n", mEnableFP16 ? "has" : "does not have");
			builder->destroy();	
		}*/
  }

  printf(LOG_GIE "device %s, %s loaded\n", deviceTypeToStr(device),
         model_path.c_str());

  /*
	 * create runtime inference engine execution context
	 */
  nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger);

  if (!infer) {
    printf(LOG_GIE "device %s, failed to create InferRuntime\n",
           deviceTypeToStr(device));
    return 0;
  }

#if NV_TENSORRT_MAJOR >= 5
#  if !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && \
        NV_TENSORRT_PATCH == 0)
  // if using DLA, set the desired core before deserialization occurs
  if (device == DEVICE_DLA_0) {
    printf(LOG_TRT "device %s, enabling DLA core 0\n", deviceTypeToStr(device));
    infer->setDLACore(0);
  } else if (device == DEVICE_DLA_1) {
    printf(LOG_TRT "device %s, enabling DLA core 1\n", deviceTypeToStr(device));
    infer->setDLACore(1);
  }
#  endif
#endif

#if NV_TENSORRT_MAJOR > 1
  // support for stringstream deserialization was deprecated in TensorRT v2
  // instead, read the stringstream into a memory buffer and pass that to TRT.
  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();
  gieModelStream.seekg(0, std::ios::beg);

  printf("MODEL SIZE: %d\n", modelSize);

  void* modelMem = malloc(modelSize);

  if (!modelMem) {
    printf(LOG_GIE "failed to allocate %i bytes to deserialize model\n",
           modelSize);
    return 0;
  }

  gieModelStream.read((char*)modelMem, modelSize);
  nvinfer1::ICudaEngine* engine =
      infer->deserializeCudaEngine(modelMem, modelSize, NULL);
  free(modelMem);
  printf(LOG_TRT "----------------------- NUMBER OF NN LAYERS: %d\n",
         engine->getNbLayers());
#else
  // TensorRT v1 can deserialize directly from stringstream
  nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream);
#endif

  if (!engine) {
    printf(LOG_GIE "device %s, failed to create CUDA engine\n",
           deviceTypeToStr(device));
    return 0;
  }

  nvinfer1::IExecutionContext* context = engine->createExecutionContext();

  if (!context) {
    printf(LOG_GIE "device %s, failed to create execution context\n",
           deviceTypeToStr(device));
    return 0;
  }

  if (mEnableDebug) {
    printf(LOG_GIE "device %s, enabling context debug sync.\n",
           deviceTypeToStr(device));
    context->setDebugSync(true);
  }

  if (mEnableProfiler) context->setProfiler(&gProfiler);

  printf(LOG_GIE
         "device %s, CUDA engine context initialized with %u bindings\n",
         deviceTypeToStr(device), engine->getNbBindings());

  mInfer = infer;
  mEngine = engine;
  mContext = context;

  SetStream(stream); // set default device stream

#if NV_TENSORRT_MAJOR >= 4
  /*
	 * print out binding info
	 */
  const int numBindings = engine->getNbBindings();

  printf(LOG_TRT "--------------------------------- GET NB BINDINGS %d\n",
         numBindings);

  for (int n = 0; n < numBindings; n++) {
    printf(LOG_TRT "binding -- index   %i\n", n);

    const char* bind_name = engine->getBindingName(n);

    printf("               -- name    '%s'\n", bind_name);
    printf("               -- type    %s\n",
           dataTypeToStr(engine->getBindingDataType(n)));
    printf("               -- in/out  %s\n",
           engine->bindingIsInput(n) ? "INPUT" : "OUTPUT");

    const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);

    printf("               -- # dims  %i\n", bind_dims.nbDims);

    for (int i = 0; i < bind_dims.nbDims; i++)
      printf("               -- dim #%i  %i (%s)\n", i, bind_dims.d[i],
             dimensionTypeToStr(bind_dims.type[i]));
  }
#endif

  /*
	 * determine dimensions of network input bindings
	 */

  std::ifstream iff(input_blob);
  printf(LOG_TRT "INPUT BLOB %s file is open: %d\n", input_blob, iff.is_open());

  const int inputIndex = engine->getBindingIndex(input_blob);

  printf(LOG_GIE "binding to input 0 %s  binding index:  %i\n", input_blob,
         inputIndex);

#if NV_TENSORRT_MAJOR > 1
  nvinfer1::Dims inputDims =
      validateDims(engine->getBindingDimensions(inputIndex));
#else
  Dims3 inputDims = engine->getBindingDimensions(inputIndex);
#endif

  size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) *
                     DIMS_W(inputDims) * sizeof(float);
  printf(LOG_GIE "binding to input 0 %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n",
         input_blob, maxBatchSize, DIMS_C(inputDims), DIMS_H(inputDims),
         DIMS_W(inputDims), inputSize);

  printf(LOG_TRT
         "------------------------------------- BINDING INPUT INDEX: %d\n",
         inputIndex);
  /*
	 * allocate memory to hold the input buffer
	 */

  printf(LOG_TRT "maxBatch: %d DIMS_C %d DIMS_H %d DIMS_W %d FLOAT: %d\n",
         maxBatchSize, DIMS_C(inputDims), DIMS_H(inputDims), DIMS_W(inputDims),
         sizeof(float));

  if (!cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize)) {
    printf(LOG_TRT
           "failed to alloc CUDA mapped memory for tensor input, %zu bytes\n",
           inputSize);
    return false;
  }

  mInputSize = inputSize;
  mWidth = DIMS_W(inputDims);
  mHeight = DIMS_H(inputDims);
  mMaxBatchSize = maxBatchSize;

  /*
	 * setup network output buffers
	 */
  const int numOutputs = output_blobs.size();

  for (int n = 0; n < numOutputs; n++) {
    const int outputIndex = engine->getBindingIndex(output_blobs[n].c_str());
    printf(LOG_GIE "binding to output %i %s  binding index:  %i\n", n,
           output_blobs[n].c_str(), outputIndex);

#if NV_TENSORRT_MAJOR > 1
    nvinfer1::Dims outputDims =
        validateDims(engine->getBindingDimensions(outputIndex));
#else
    Dims3 outputDims = engine->getBindingDimensions(outputIndex);
#endif

    size_t outputSize = maxBatchSize * DIMS_C(outputDims) * DIMS_H(outputDims) *
                        DIMS_W(outputDims) * sizeof(float);
    printf(LOG_GIE
           "binding to output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n",
           n, output_blobs[n].c_str(), maxBatchSize, DIMS_C(outputDims),
           DIMS_H(outputDims), DIMS_W(outputDims), outputSize);

    // allocate output memory
    void* outputCPU = NULL;
    void* outputCUDA = NULL;

    //if( CUDA_FAILED(cudaMalloc((void**)&outputCUDA, outputSize)) )
    if (!cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize)) {
      printf(
          LOG_TRT
          "failed to alloc CUDA mapped memory for tensor output, %zu bytes\n",
          outputSize);
      return false;
    }

    outputLayer l;

    l.CPU = (float*)outputCPU;
    l.CUDA = (float*)outputCUDA;
    l.size = outputSize;

#if NV_TENSORRT_MAJOR > 1
    DIMS_W(l.dims) = DIMS_W(outputDims);
    DIMS_H(l.dims) = DIMS_H(outputDims);
    DIMS_C(l.dims) = DIMS_C(outputDims);
#else
    l.dims = outputDims;
#endif

    l.name = output_blobs[n];
    mOutputs.push_back(l);
  }

#if NV_TENSORRT_MAJOR > 1
  DIMS_W(mInputDims) = DIMS_W(inputDims);
  DIMS_H(mInputDims) = DIMS_H(inputDims);
  DIMS_C(mInputDims) = DIMS_C(inputDims);
#else
  mInputDims = inputDims;
#endif
  mPrototxtPath = prototxt_path;
  mModelPath = model_path;
  mInputBlobName = input_blob;
  mPrecision = precision;
  mDevice = device;
  mAllowGPUFallback = allowGPUFallback;

  if (mean_path != NULL) mMeanPath = mean_path;

  printf("device %s, %s initialized.\n", deviceTypeToStr(device),
         mModelPath.c_str());
  return true;
}

// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path_, const char* model_path_, const char* mean_path, 
					    const char* input_blob, const Dims3& input_dims,
					    const std::vector<std::string>& output_blobs, 
					    uint32_t maxBatchSize, precisionType precision,
				   	    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
	if( /*!prototxt_path_ ||*/ !model_path_ )
		return false;

#if NV_TENSORRT_MAJOR >= 4
	printf(LOG_TRT "TensorRT version %u.%u.%u\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
#else
	printf(LOG_TRT "TensorRT version %u.%u\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
#endif

	/*
	 * load NV inference plugins
	 */
#if NV_TENSORRT_MAJOR > 4
	static bool loadedPlugins = false;

	if( !loadedPlugins )
	{
		printf(LOG_TRT "loading NVIDIA plugins...\n");

		loadedPlugins = initLibNvInferPlugins(&gLogger, "");

		if( !loadedPlugins )
			printf(LOG_TRT "failed to load NVIDIA plugins\n");
		else
			printf(LOG_TRT "completed loading NVIDIA plugins.\n");
	}
#endif

	/*
	 * verify the prototxt and model paths
	 */
	const std::string model_path    = locateFile(model_path_);
	const std::string prototxt_path = locateFile(prototxt_path_ != NULL ? prototxt_path_ : "");
	
	const std::string model_ext = fileExtension(model_path_);
	const modelFormat   model_fmt = modelFormatFromStr(model_ext.c_str());

	printf(LOG_TRT "detected model format - %s  (extension '.%s')\n", modelFormatToStr(model_fmt), model_ext.c_str());

	if( model_fmt == MODEL_CUSTOM )
	{
		printf(LOG_TRT "model format '%s' not supported by jetson-inference\n", modelFormatToStr(model_fmt));
		return false;
	}
#if NV_TENSORRT_MAJOR < 5
	else if( model_fmt == MODEL_ONNX )
	{
		printf(LOG_TRT "importing ONNX models is not supported in TensorRT %u.%u (version >= 5.0 required)\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
	else if( model_fmt == MODEL_UFF )
	{
		printf(LOG_TRT "importing UFF models is not supported in TensorRT %u.%u (version >= 5.0 required)\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
#endif
	else if( model_fmt == MODEL_CAFFE && !prototxt_path_ )
	{
		printf(LOG_TRT "attempted to load caffe model without specifying prototxt file\n");
		return false;
	}

	mModelFormat = model_fmt;


	/*
	 * if the precision is left unspecified, detect the fastest
	 */
	printf(LOG_TRT "desired precision specified for %s: %s\n", deviceTypeToStr(device), precisionTypeToStr(precision));

	if( precision == TYPE_DISABLED )
	{
		printf(LOG_TRT "skipping network specified with precision TYPE_DISABLE\n");
		printf(LOG_TRT "please specify a valid precision to create the network\n");

		return false;
	}
	else if( precision == TYPE_FASTEST )
	{
		if( !calibrator )
			printf(LOG_TRT "requested fasted precision for device %s without providing valid calibrator, disabling INT8\n", deviceTypeToStr(device));

		precision = FindFastestPrecision(device, (calibrator != NULL));
		printf(LOG_TRT "selecting fastest native precision for %s:  %s\n", deviceTypeToStr(device), precisionTypeToStr(precision));
	}
	else
	{
		if( !DetectNativePrecision(precision, device) )
		{
			printf(LOG_TRT "precision %s is not supported for device %s\n", precisionTypeToStr(precision), deviceTypeToStr(device));
			return false;
		}

		if( precision == TYPE_INT8 && !calibrator )
			printf(LOG_TRT "warning:  device %s using INT8 precision with RANDOM calibration\n", deviceTypeToStr(device));
	}


	/*
	 * attempt to load network from cache before profiling with tensorRT
	 */
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);

	char cache_prefix[512];
	char cache_path[512];

	sprintf(cache_prefix, "%s.%u.%u.%i.%s.%s", model_path.c_str(), maxBatchSize, (uint32_t)allowGPUFallback, NV_TENSORRT_VERSION, deviceTypeToStr(device), precisionTypeToStr(precision));
	sprintf(cache_path, "%s.calibration", cache_prefix);
	mCacheCalibrationPath = cache_path;
	
	sprintf(cache_path, "%s.engine", cache_prefix);
	mCacheEnginePath = cache_path;	
	printf(LOG_TRT "attempting to open engine cache file %s\n", mCacheEnginePath.c_str());
	
	std::ifstream cache( mCacheEnginePath );

	if( !cache )
	{
		printf(LOG_TRT "cache file not found, profiling network model on device %s\n", deviceTypeToStr(device));
	
		if( model_path.size() == 0 )
		{
			printf("\nerror:  model file '%s' was not found.\n", model_path_);
			printf("%s\n", LOG_DOWNLOADER_TOOL);
			return 0;
		}

		if( !ProfileModel(prototxt_path, model_path, input_blob, input_dims,
						 output_blobs, maxBatchSize, precision, device, 
						 allowGPUFallback, calibrator, gieModelStream) )
		{
			printf(LOG_TRT "device %s, failed to load %s\n", deviceTypeToStr(device), model_path_);
			return 0;
		}
	
		printf(LOG_TRT "network profiling complete, writing engine cache to %s\n", mCacheEnginePath.c_str());
		std::ofstream outFile;
		outFile.open(mCacheEnginePath);
		outFile << gieModelStream.rdbuf();
		outFile.close();
		gieModelStream.seekg(0, gieModelStream.beg);
		printf(LOG_TRT "device %s, completed writing engine cache to %s\n", deviceTypeToStr(device), mCacheEnginePath.c_str());
	}
	else
	{
		printf(LOG_TRT "loading network profile from engine cache... %s\n", mCacheEnginePath.c_str());
		gieModelStream << cache.rdbuf();
		cache.close();

		// test for half FP16 support
		// nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
		
		// if( builder != NULL )
		// {
		// 	mEnableFP16 = !mOverride16 && builder->platformHasFastFp16();
		// 	printf(LOG_TRT "platform %s fast FP16 support\n", mEnableFP16 ? "has" : "does not have");
		// 	builder->destroy();	
		// }
	}

	printf(LOG_TRT "device %s, %s loaded\n", deviceTypeToStr(device), model_path.c_str());
	

	/*
	 * create runtime inference engine execution context
	 */
	nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger);
	
	if( !infer )
	{
		printf(LOG_TRT "device %s, failed to create InferRuntime\n", deviceTypeToStr(device));
		return 0;
	}

#if NV_TENSORRT_MAJOR >= 5 
#if !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0)
	// if using DLA, set the desired core before deserialization occurs
	if( device == DEVICE_DLA_0 )
	{
		printf(LOG_TRT "device %s, enabling DLA core 0\n", deviceTypeToStr(device));
		infer->setDLACore(0);
	}
	else if( device == DEVICE_DLA_1 )
	{
		printf(LOG_TRT "device %s, enabling DLA core 1\n", deviceTypeToStr(device));
		infer->setDLACore(1);
	}
#endif
#endif

#if NV_TENSORRT_MAJOR > 1
	// support for stringstream deserialization was deprecated in TensorRT v2
	// instead, read the stringstream into a memory buffer and pass that to TRT.
	gieModelStream.seekg(0, std::ios::end);
	const int modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, std::ios::beg);

	void* modelMem = malloc(modelSize);

	if( !modelMem )
	{
		printf(LOG_TRT "failed to allocate %i bytes to deserialize model\n", modelSize);
		return 0;
	}

	gieModelStream.read((char*)modelMem, modelSize);
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(modelMem, modelSize, NULL);
	free(modelMem);
#else
	// TensorRT v1 can deserialize directly from stringstream
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream);
#endif

	if( !engine )
	{
		printf(LOG_TRT "device %s, failed to create CUDA engine\n", deviceTypeToStr(device));
		return 0;
	}
	
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	
	if( !context )
	{
		printf(LOG_TRT "device %s, failed to create execution context\n", deviceTypeToStr(device));
		return 0;
	}

	if( mEnableDebug )
	{
		printf(LOG_TRT "device %s, enabling context debug sync.\n", deviceTypeToStr(device));
		context->setDebugSync(true);
	}

	if( mEnableProfiler )
		context->setProfiler(&gProfiler);

	printf(LOG_TRT "device %s, CUDA engine context initialized with %u bindings\n", deviceTypeToStr(device), engine->getNbBindings());
	
	mInfer   = infer;
	mEngine  = engine;
	mContext = context;
	
	SetStream(stream);	// set default device stream


#if NV_TENSORRT_MAJOR >= 4
	/*
	 * print out binding info
	 */
	const int numBindings = engine->getNbBindings();
	
	for( int n=0; n < numBindings; n++ )
	{
		printf(LOG_TRT "binding -- index   %i\n", n);

		const char* bind_name = engine->getBindingName(n);

		printf("               -- name    '%s'\n", bind_name);
		printf("               -- type    %s\n", dataTypeToStr(engine->getBindingDataType(n)));
		printf("               -- in/out  %s\n", engine->bindingIsInput(n) ? "INPUT" : "OUTPUT");

		const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);

		printf("               -- # dims  %i\n", bind_dims.nbDims);
		
		for( int i=0; i < bind_dims.nbDims; i++ )
			printf("               -- dim #%i  %i (%s)\n", i, bind_dims.d[i], dimensionTypeToStr(bind_dims.type[i]));
	}
#endif

	/*
	 * determine dimensions of network input bindings
	 */
	const int inputIndex = engine->getBindingIndex(input_blob);
	
	printf(LOG_TRT "binding to input 0 %s  binding index:  %i\n", input_blob, inputIndex);
	
#if NV_TENSORRT_MAJOR > 1
	nvinfer1::Dims inputDims = validateDims(engine->getBindingDimensions(inputIndex));

#if NV_TENSORRT_MAJOR >= 7
    if( mModelType == MODEL_ONNX )
        inputDims = shiftDims(inputDims);   // change NCHW to CHW if EXPLICIT_BATCH set
#endif
#else
    Dims3 inputDims = engine->getBindingDimensions(inputIndex);
#endif

	size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);
	printf(LOG_TRT "binding to input 0 %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", input_blob, maxBatchSize, DIMS_C(inputDims), DIMS_H(inputDims), DIMS_W(inputDims), inputSize);
	

	/*
	 * allocate memory to hold the input buffer
	 */
	if( !cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize) )
	{
		printf(LOG_TRT "failed to alloc CUDA mapped memory for tensor input, %zu bytes\n", inputSize);
		return false;
	}
	
	mInputSize    = inputSize;
	mWidth        = DIMS_W(inputDims);
	mHeight       = DIMS_H(inputDims);
	mMaxBatchSize = maxBatchSize;
	

	/*
	 * setup network output buffers
	 */
	const int numOutputs = output_blobs.size();
	
	for( int n=0; n < numOutputs; n++ )
	{
		const int outputIndex = engine->getBindingIndex(output_blobs[n].c_str());
		printf(LOG_TRT "binding to output %i %s  binding index:  %i\n", n, output_blobs[n].c_str(), outputIndex);

	#if NV_TENSORRT_MAJOR > 1
		nvinfer1::Dims outputDims = validateDims(engine->getBindingDimensions(outputIndex));

    #if NV_TENSORRT_MAJOR >= 7
        if( mModelType == MODEL_ONNX )
            outputDims = shiftDims(outputDims);  // change NCHW to CHW if EXPLICIT_BATCH set
    #endif
	#else
		Dims3 outputDims = engine->getBindingDimensions(outputIndex);
	#endif

		size_t outputSize = maxBatchSize * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);
		printf(LOG_TRT "binding to output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", n, output_blobs[n].c_str(), maxBatchSize, DIMS_C(outputDims), DIMS_H(outputDims), DIMS_W(outputDims), outputSize);
	
		// allocate output memory 
		void* outputCPU  = NULL;
		void* outputCUDA = NULL;
		
		//if( CUDA_FAILED(cudaMalloc((void**)&outputCUDA, outputSize)) )
		if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
		{
			printf(LOG_TRT "failed to alloc CUDA mapped memory for tensor output, %zu bytes\n", outputSize);
			return false;
		}
	
		outputLayer l;
		
		l.CPU  = (float*)outputCPU;
		l.CUDA = (float*)outputCUDA;
		l.size = outputSize;

	#if NV_TENSORRT_MAJOR > 1
		DIMS_W(l.dims) = DIMS_W(outputDims);
		DIMS_H(l.dims) = DIMS_H(outputDims);
		DIMS_C(l.dims) = DIMS_C(outputDims);
	#else
		l.dims = outputDims;
	#endif

		l.name = output_blobs[n];
		mOutputs.push_back(l);
	}
	

	/*
	 * create events for timing
	 */
	for( int n=0; n < PROFILER_TOTAL * 2; n++ )
		CUDA(cudaEventCreate(&mEventsGPU[n]));


#if NV_TENSORRT_MAJOR > 1
	DIMS_W(mInputDims) = DIMS_W(inputDims);
	DIMS_H(mInputDims) = DIMS_H(inputDims);
	DIMS_C(mInputDims) = DIMS_C(inputDims);
#else
	mInputDims        = inputDims;
#endif
	mPrototxtPath     = prototxt_path;
	mModelPath        = model_path;
	mInputBlobName    = input_blob;
	mPrecision        = precision;
	mDevice           = device;
	mAllowGPUFallback = allowGPUFallback;

	if( mean_path != NULL )
		mMeanPath = mean_path;
	
	printf("device %s, %s initialized.\n", deviceTypeToStr(device), mModelPath.c_str());
	return true;
}

// CreateStream
cudaStream_t tensorNet::CreateStream(bool nonBlocking) {
  uint32_t flags = cudaStreamDefault;

  if (nonBlocking) flags = cudaStreamNonBlocking;

  cudaStream_t stream = NULL;

  if (CUDA_FAILED(cudaStreamCreateWithFlags(&stream, flags))) return NULL;

  SetStream(stream);
  return stream;
}

// SetStream
void tensorNet::SetStream(cudaStream_t stream) {
  mStream = stream;

  if (!mStream) return;

  for (int n = 0; n < 2; n++) {
    if (!mEvents[n])
      CUDA(cudaEventCreateWithFlags(
          &mEvents[n], /*cudaEventBlockingSync*/ cudaEventDefault));
  }
}


using namespace nvinfer1;
using namespace nvuffparser;

class FlattenConcat : public IPluginV2
{
public:
    FlattenConcat(int concatAxis, bool ignoreBatch)
        : mIgnoreBatch(ignoreBatch)
        , mConcatAxisID(concatAxis)
    {
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    }
    //clone constructor
    FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis, int* inputConcatAxis)
        : mIgnoreBatch(ignoreBatch)
        , mConcatAxisID(concatAxis)
        , mOutputConcatAxis(outputConcatAxis)
        , mNumInputs(numInputs)
    {
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        for (int i = 0; i < mNumInputs; ++i)
            mInputConcatAxis[i] = inputConcatAxis[i];
    }

    FlattenConcat(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        mIgnoreBatch = read<bool>(d);
        mConcatAxisID = read<int>(d);
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
        mOutputConcatAxis = read<int>(d);
        mNumInputs = read<int>(d);
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        CHECK(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

        std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

        mCHW = read<nvinfer1::DimsCHW>(d);

        std::for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

        assert(d == a + length);
    }
    ~FlattenConcat()
    {
        if (mInputConcatAxis)
            CHECK(cudaFreeHost(mInputConcatAxis));
        if (mCopySize)
            CHECK(cudaFreeHost(mCopySize));
    }
    int getNbOutputs() const override { return 1; }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims >= 1);
        assert(index == 0);
        mNumInputs = nbInputDims;
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        mOutputConcatAxis = 0;

        for (int i = 0; i < nbInputDims; ++i)
        {
            int flattenInput = 0;
            assert(inputs[i].nbDims == 3);
            if (mConcatAxisID != 1)
                assert(inputs[i].d[0] == inputs[0].d[0]);
            if (mConcatAxisID != 2)
                assert(inputs[i].d[1] == inputs[0].d[1]);
            if (mConcatAxisID != 3)
                assert(inputs[i].d[2] == inputs[0].d[2]);
            flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            mInputConcatAxis[i] = flattenInput;
            mOutputConcatAxis += mInputConcatAxis[i];
        }

        return DimsCHW(mConcatAxisID == 1 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 2 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 3 ? mOutputConcatAxis : 1);
    }

    int initialize() override
    {
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    void terminate() override
    {
        CHECK(cublasDestroy(mCublas));
    }

    size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
    {
        int numConcats = 1;
        assert(mConcatAxisID != 0);
        numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());
        cublasSetStream(mCublas, stream);

        if (!mIgnoreBatch)
            numConcats *= batchSize;

        float* output = reinterpret_cast<float*>(outputs[0]);
        int offset = 0;
        for (int i = 0; i < mNumInputs; ++i)
        {
            const float* input = reinterpret_cast<const float*>(inputs[i]);
            float* inputTemp;
            CHECK(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

            CHECK(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

            for (int n = 0; n < numConcats; ++n)
            {
                CHECK(cublasScopy(mCublas, mInputConcatAxis[i],
                                  inputTemp + n * mInputConcatAxis[i], 1,
                                  output + (n * mOutputConcatAxis + offset), 1));
            }
            CHECK(cudaFree(inputTemp));
            offset += mInputConcatAxis[i];
        }

        return 0;
    }

    size_t getSerializationSize() const override
    {
        return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
    }

    void serialize(void* buffer) const override
    {
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, mIgnoreBatch);
        write(d, mConcatAxisID);
        write(d, mOutputConcatAxis);
        write(d, mNumInputs);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mInputConcatAxis[i]);
        }
        write(d, mCHW);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mCopySize[i]);
        }
        assert(d == a + getSerializationSize());
    }

    void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override
    {
        assert(nbOutputs == 1);
        mCHW = inputs[0];
        assert(inputs[0].nbDims == 3);
        CHECK(cudaMallocHost((void**) &mCopySize, nbInputs * sizeof(int)));
        for (int i = 0; i < nbInputs; ++i)
        {
            mCopySize[i] = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
        }
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    }
    const char* getPluginType() const override { return "FlattenConcat_TRT"; }

    const char* getPluginVersion() const override { return "1"; }

    void destroy() override { delete this; }

    IPluginV2* clone() const override
    {
        return new FlattenConcat(mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis, mInputConcatAxis);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace()  const override { return mNamespace.c_str(); }

private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    size_t* mCopySize = nullptr;
    bool mIgnoreBatch{false};
    int mConcatAxisID{0}, mOutputConcatAxis{0}, mNumInputs{0};
    int* mInputConcatAxis = nullptr;
    nvinfer1::Dims mCHW;
    cublasHandle_t mCublas;
    std::string mNamespace;
};

namespace
{
const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};
} // namespace

class FlattenConcatPluginCreator : public IPluginCreator
{
public:
    FlattenConcatPluginCreator()
    {
        mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~FlattenConcatPluginCreator() {}

    const char* getPluginName() const override { return FLATTENCONCAT_PLUGIN_NAME; }

    const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "axis"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                mConcatAxisID = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "ignoreBatch"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
            }
        }

        return new FlattenConcat(mConcatAxisID, mIgnoreBatch);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {

        //This object will be deleted when the network is destroyed, which will
        //call Concat::destroy()
        return new FlattenConcat(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    static PluginFieldCollection mFC;
    bool mIgnoreBatch{false};
    int mConcatAxisID;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace = "";
};

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FlattenConcatPluginCreator);