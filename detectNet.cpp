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
 
#include "detectNet.h"
#include "imageNet.cuh"

#include "cudaMappedMemory.h"
#include "cudaFont.h"

#include "cudaOverlay.h"
#include "cudaResize.h"

#include "commandLine.h"
#include "filesystem.h"
#include <iostream>

#define OUTPUT_CVG  0
#define OUTPUT_BBOX 1

#define OUTPUT_UFF  0	// UFF has primary output containing detection results
#define OUTPUT_NUM	1	// UFF has secondary output containing one detection count
//#define DEBUG_CLUSTERING

#define CHECK_NULL_STR(x)	(x != NULL) ? x : "NULL"



// constructor
detectNet::detectNet() : tensorNet()
{
	mCoverageThreshold = 0.5f;
	mMeanPixel         = 0.0f;
	mCustomClasses     = 0;

	mClassColors[0] = NULL;	// cpu ptr
	mClassColors[1] = NULL; // gpu ptr
}

// constructor
detectNet::detectNet( float meanPixel ) : tensorNet()
{
	mCoverageThreshold = DETECTNET_DEFAULT_THRESHOLD;
	mMeanPixel         = meanPixel;
	mCustomClasses     = 0;
	mNumClasses        = 0;

	mClassColors[0]   = NULL; // cpu ptr
	mClassColors[1]   = NULL; // gpu ptr
	
	mDetectionSets[0] = NULL; // cpu ptr
	mDetectionSets[1] = NULL; // gpu ptr
	mDetectionSet     = 0;
	mMaxDetections    = 0;
}



// destructor
detectNet::~detectNet()
{
	if( mDetectionSets != NULL )
	{
		CUDA(cudaFreeHost(mDetectionSets[0]));
		
		mDetectionSets[0] = NULL;
		mDetectionSets[1] = NULL;
	}
	
	if( mClassColors != NULL )
	{
		CUDA(cudaFreeHost(mClassColors[0]));
		
		mClassColors[0] = NULL;
		mClassColors[1] = NULL;
	}
}


// Create
detectNet* detectNet::Create( const char* prototxt, const char* model, float mean_pixel, const char* class_labels,
						float threshold, const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
						uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	detectNet* net = new detectNet(0.0f);
	
	if( !net )
		return NULL;

	printf("\n");
	printf("detectNet -- loading detection network model from:\n");
	printf("          -- prototxt     %s\n", prototxt);
	printf("          -- model        %s\n", model);
	printf("          -- input_blob   '%s'\n", input_blob);
	printf("          -- output_cvg   '%s'\n", coverage_blob);
	printf("          -- output_bbox  '%s'\n", bbox_blob);
	printf("          -- mean_pixel   %f\n", mean_pixel);
	printf("          -- class_labels %s\n", (class_labels != NULL) ? class_labels : "NULL");
	printf("          -- threshold    %f\n", threshold);
	printf("          -- batch_size   %u\n\n", maxBatchSize);
	
	//net->EnableDebug();
	
	std::vector<std::string> output_blobs;
	output_blobs.push_back(coverage_blob);
	output_blobs.push_back(bbox_blob);
	
	// load the model
	if( !net->LoadNetwork(prototxt, model, NULL, input_blob, output_blobs, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("detectNet -- failed to initialize.\n");
		return NULL;
	}
	
	// set default class colors
	if( !net->defaultColors() )
		return NULL;
	
	// load class descriptions
	net->loadClassDesc(class_labels);
	net->defaultClassDesc();

	// set the threshold & mean pixel
	net->SetThreshold(threshold);
	net->mMeanPixel = mean_pixel;

	// get the maximum bounding boxes
	printf("detectNet -- maximum bounding boxes:  %u\n", net->GetMaxBoundingBoxes());
	return net;
}



// Create


detectNet* detectNet::Create( const char* prototxt, const char* model, const char* mean_binary, const char* class_labels, 
						float threshold, const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
						uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{

	detectNet* net = new detectNet(0.0f);
	
	if( !net )
		return NULL;

	printf("\n");
	printf("detectNet -- loading detection network model from:\n");
	printf("          -- prototxt    %s\n", prototxt);
	printf("          -- model       %s\n", model);
	printf("          -- input_blob  '%s'\n", input_blob);
	printf("          -- output_cvg  '%s'\n", coverage_blob);
	printf("          -- output_bbox '%s'\n", bbox_blob);
	printf("          -- mean_binary  %s\n", (mean_binary != NULL) ? mean_binary : "NULL");
	printf("          -- class_labels %s\n", (class_labels != NULL) ? class_labels : "NULL");
	printf("          -- threshold    %f\n", threshold);
	printf("          -- batch_size   %u\n\n", maxBatchSize);
	
	//net->EnableDebug();
	
	std::vector<std::string> output_blobs;
	output_blobs.push_back(coverage_blob);
	output_blobs.push_back(bbox_blob);
	
	// load the model
	if( !net->LoadNetwork(prototxt, model, mean_binary, input_blob, output_blobs, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("detectNet -- failed to initialize.\n");
		return NULL;
	}
	
	// set default class colors
	if( !net->defaultColors() )
		return NULL;

	// load class descriptions
	net->loadClassDesc(class_labels);
	net->defaultClassDesc();
	
	// set the specified threshold
	net->SetThreshold(threshold);

	// get the maximum bounding boxes
	printf("detectNet -- maximum bounding boxes:  %u\n", net->GetMaxBoundingBoxes());
	return net;
}


// defaultColors
bool detectNet::defaultColors()
{
	const uint32_t numClasses = GetNumClasses();
	
	if( !cudaAllocMapped((void**)&mClassColors[0], (void**)&mClassColors[1], numClasses * sizeof(float4)) )
		return false;
	
// if there are a large number of classes (MS COCO)
	// programatically generate the class color map
	if( numClasses > 10 )
	{
		// https://github.com/dusty-nv/pytorch-segmentation/blob/16882772bc767511d892d134918722011d1ea771/datasets/sun_remap.py#L90
		#define bitget(byteval, idx)	((byteval & (1 << idx)) != 0)

		for( int i=0; i < numClasses; i++ )
		{
			int r = 0;
			int g = 0;
			int b = 0;
			int c = i;

			for( int j=0; j < 8; j++ )
			{
				r = r | (bitget(c, 0) << 7 - j);
				g = g | (bitget(c, 1) << 7 - j);
				b = b | (bitget(c, 2) << 7 - j);
				c = c >> 3;
			}

			mClassColors[0][i*4+0] = r;
			mClassColors[0][i*4+1] = g;
			mClassColors[0][i*4+2] = b;
			mClassColors[0][i*4+3] = DETECTNET_DEFAULT_ALPHA; 

			//printf(LOG_TRT "color %02i  %3i %3i %3i %3i\n", i, (int)r, (int)g, (int)b, (int)alpha);
		}
	}
	else
	{
		// blue colors, except class 1 is green
		for( uint32_t n=0; n < numClasses; n++ )
		{
			if( n != 1 )
			{
				mClassColors[0][n*4+0] = 0.0f;	// r
				mClassColors[0][n*4+1] = 200.0f;	// g
				mClassColors[0][n*4+2] = 255.0f;	// b
				mClassColors[0][n*4+3] = DETECTNET_DEFAULT_ALPHA;	// a
			}
			else
			{
				mClassColors[0][n*4+0] = 0.0f;	// r
				mClassColors[0][n*4+1] = 255.0f;	// g
				mClassColors[0][n*4+2] = 175.0f;	// b
				mClassColors[0][n*4+3] = 75.0f;	// a
			}
		}
	}

	return true;
}


// defaultClassDesc
void detectNet::defaultClassDesc()
{
	const uint32_t numClasses = GetNumClasses();
	const int syn = 9;  // length of synset prefix (in characters)
	
	// assign defaults to classes that have no info
	for( uint32_t n=mClassDesc.size(); n < numClasses; n++ )
	{
		char syn_str[10];
		sprintf(syn_str, "n%08u", mCustomClasses);

		char desc_str[16];
		sprintf(desc_str, "class #%u", mCustomClasses);

		mClassSynset.push_back(syn_str);
		mClassDesc.push_back(desc_str);

		mCustomClasses++;
	}
}


// loadClassDesc
bool detectNet::loadClassDesc( const char* filename )
{
	printf("detectNet -- model has %u object classes\n", GetNumClasses());

	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		printf("detectNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		printf("detectNet -- failed to open %s\n", path.c_str());
		return false;
	}
	
	// read class descriptions
	char str[512];

	while( fgets(str, 512, f) != NULL )
	{
		const int syn = 9;  // length of synset prefix (in characters)
		const int len = strlen(str);
		
		if( len > syn && str[0] == 'n' && str[syn] == ' ' )
		{
			str[syn]   = 0;
			str[len-1] = 0;
	
			const std::string a = str;
			const std::string b = (str + syn + 1);
	
			//printf("a=%s b=%s\n", a.c_str(), b.c_str());

			mClassSynset.push_back(a);
			mClassDesc.push_back(b);
		}
		else if( len > 0 )	// no 9-character synset prefix (i.e. from DIGITS snapshot)
		{
			char a[10];
			sprintf(a, "n%08u", mCustomClasses);

			//printf("a=%s b=%s (custom non-synset)\n", a, str);
			mCustomClasses++;

			if( str[len-1] == '\n' )
				str[len-1] = 0;

			mClassSynset.push_back(a);
			mClassDesc.push_back(str);
		}
	}
	
	fclose(f);
	
	printf("detectNet -- loaded %zu class info entries\n", mClassSynset.size());
	
	if( mClassSynset.size() == 0 )
		return false;

	if( IsModelType(MODEL_UFF) )
		mNumClasses = mClassDesc.size();

	printf("detectNet -- number of object classes:  %u\n", mNumClasses);
	mClassPath = path;	
	return true;
}



// Create
detectNet* detectNet::Create( NetworkType networkType, float threshold, uint32_t maxBatchSize, 
						precisionType precision, deviceType device, bool allowGPUFallback )
{
#if 1
	if( networkType == PEDNET_MULTI )
		return Create("third_party/jetson-inference/data/multiped-500/deploy.prototxt", 
		"third_party/jetson-inference/data/multiped-500/snapshot_iter_178000.caffemodel", 117.0f, 
		"third_party/jetson-inference/data/multiped-500/class_labels.txt", 
		threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == FACENET )
		return Create("third_party/jetson-inference/data/facenet-120/deploy.prototxt", "third_party/jetson-inference/data/facenet-120/snapshot_iter_24000.caffemodel", 0.0f, "third_party/jetson-inference/data/facenet-120/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == PEDNET )
		return Create("third_party/jetson-inference/data/ped-100/deploy.prototxt", "third_party/jetson-inference/data/ped-100/snapshot_iter_70800.caffemodel", 0.0f, "third_party/jetson-inference/data/ped-100/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_AIRPLANE )
		return Create("third_party/jetson-inference/data/DetectNet-COCO-Airplane/deploy.prototxt", "third_party/jetson-inference/data/DetectNet-COCO-Airplane/snapshot_iter_22500.caffemodel", 0.0f, "third_party/jetson-inference/data/DetectNet-COCO-Airplane/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_BOTTLE )
		return Create("third_party/jetson-inference/data/DetectNet-COCO-Bottle/deploy.prototxt", "third_party/jetson-inference/data/DetectNet-COCO-Bottle/snapshot_iter_59700.caffemodel", 0.0f, "third_party/jetson-inference/data/DetectNet-COCO-Bottle/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_CHAIR )
		return Create("third_party/jetson-inference/data/DetectNet-COCO-Chair/deploy.prototxt", "third_party/jetson-inference/data/DetectNet-COCO-Chair/snapshot_iter_89500.caffemodel", 0.0f, "third_party/jetson-inference/data/DetectNet-COCO-Chair/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_DOG )
		return Create("third_party/jetson-inference/data/DetectNet-COCO-Dog/deploy.prototxt", "third_party/jetson-inference/data/DetectNet-COCO-Dog/snapshot_iter_38600.caffemodel", 0.0f, "third_party/jetson-inference/data/DetectNet-COCO-Dog/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
#if NV_TENSORRT_MAJOR > 4
	else if( networkType == SSD_INCEPTION_V2 )
		return Create("/home/one/nautilus/third_party/jetson-inference/data/SSD-Inception-v2/ssd_inception_v2_coco.uff", "/home/one/nautilus/third_party/jetson-inference/data/SSD-Inception-v2/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "NMS", "NMS_1", maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == SSD_MOBILENET_V1 )
		return Create("/home/one/nautilus/third_party/jetson-inference/data/SSD-Mobilenet-v1/ssd_mobilenet_v1_coco.uff", "/home/one/nautilus/third_party/jetson-inference/data/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "Postprocessor", "Postprocessor_1", maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == SSD_MOBILENET_V2 )
		return Create("/home/one/nautilus/third_party/jetson-inference/data/ssd_mobilenet_v2_coco.uff", "/home/one/nautilus/third_party/jetson-inference/data/SSD-Mobilenet-v2/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "NMS", "NMS_1", maxBatchSize, precision, device, allowGPUFallback);
#endif
	else
		return NULL;
#else
#endif
}


// NetworkTypeFromStr
detectNet::NetworkType detectNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return detectNet::CUSTOM;

	detectNet::NetworkType type = detectNet::PEDNET;

	if( strcasecmp(modelName, "multiped") == 0 || strcasecmp(modelName, "multiped-500") == 0 )
		type = detectNet::PEDNET_MULTI;
	else if( strcasecmp(modelName, "pednet") == 0 || strcasecmp(modelName, "ped-100") == 0 )
		type = detectNet::PEDNET;
	else if( strcasecmp(modelName, "facenet") == 0 || strcasecmp(modelName, "facenet-120") == 0 || strcasecmp(modelName, "face-120") == 0 )
		type = detectNet::FACENET;
	else if( strcasecmp(modelName, "coco-airplane") == 0 || strcasecmp(modelName, "airplane") == 0 )
		type = detectNet::COCO_AIRPLANE;
	else if( strcasecmp(modelName, "coco-bottle") == 0 || strcasecmp(modelName, "bottle") == 0 )
		type = detectNet::COCO_BOTTLE;
	else if( strcasecmp(modelName, "coco-chair") == 0 || strcasecmp(modelName, "chair") == 0 )
		type = detectNet::COCO_CHAIR;
	else if( strcasecmp(modelName, "coco-dog") == 0 || strcasecmp(modelName, "dog") == 0 )
		type = detectNet::COCO_DOG;
#if NV_TENSORRT_MAJOR > 4
	else if( strcasecmp(modelName, "ssd-inception") == 0 || strcasecmp(modelName, "ssd-inception-v2") == 0 || strcasecmp(modelName, "coco-ssd-inception") == 0 || strcasecmp(modelName, "coco-ssd-inception-v2") == 0)
		type = detectNet::SSD_INCEPTION_V2;
	else if( strcasecmp(modelName, "ssd-mobilenet-v1") == 0 || strcasecmp(modelName, "coco-ssd-mobilenet-v1") == 0)
		type = detectNet::SSD_MOBILENET_V1;
	else if( strcasecmp(modelName, "ssd-mobilenet-v2") == 0 || strcasecmp(modelName, "coco-ssd-mobilenet-v2") == 0 || strcasecmp(modelName, "ssd-mobilenet") == 0 )
		type = detectNet::SSD_MOBILENET_V2;
#endif
	else
		type = detectNet::CUSTOM;

	return type;
}


// Create

detectNet* detectNet::Create( int argc, char** argv )
{
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("model");

	if( !modelName )
	{
		if( argc == 2 )
			modelName = argv[1];
		else if( argc == 4 )
			modelName = argv[3];
		else
			modelName = "pednet";
	}

	//if( argc > 3 )
	//	modelName = argv[3];	

	detectNet::NetworkType type = detectNet::PEDNET_MULTI;

	if( strcasecmp(modelName, "multiped") == 0 || strcasecmp(modelName, "multiped-500") == 0 )
		type = detectNet::PEDNET_MULTI;
	else if( strcasecmp(modelName, "pednet") == 0 || strcasecmp(modelName, "ped-100") == 0 )
		type = detectNet::PEDNET;
	else if( strcasecmp(modelName, "facenet") == 0 || strcasecmp(modelName, "facenet-120") == 0 || strcasecmp(modelName, "face-120") == 0 )
		type = detectNet::FACENET;
	else if( strcasecmp(modelName, "coco-airplane") == 0 || strcasecmp(modelName, "airplane") == 0 )
		type = detectNet::COCO_AIRPLANE;
	else if( strcasecmp(modelName, "coco-bottle") == 0 || strcasecmp(modelName, "bottle") == 0 )
		type = detectNet::COCO_BOTTLE;
	else if( strcasecmp(modelName, "coco-chair") == 0 || strcasecmp(modelName, "chair") == 0 )
		type = detectNet::COCO_CHAIR;
	else if( strcasecmp(modelName, "coco-dog") == 0 || strcasecmp(modelName, "dog") == 0 )
		type = detectNet::COCO_DOG;
	else
	{
		const char* prototxt     = cmdLine.GetString("prototxt");
		const char* input        = cmdLine.GetString("input_blob");
		const char* out_cvg      = cmdLine.GetString("output_cvg");
		const char* out_bbox     = cmdLine.GetString("output_bbox");
		const char* class_labels = cmdLine.GetString("class_labels");

		if( !input ) 	input     = DETECTNET_DEFAULT_INPUT;
		if( !out_cvg )  out_cvg  = DETECTNET_DEFAULT_COVERAGE;
		if( !out_bbox ) out_bbox = DETECTNET_DEFAULT_BBOX;
		
		float meanPixel = cmdLine.GetFloat("mean_pixel");
		float threshold = cmdLine.GetFloat("threshold");
		
		if( threshold == 0.0f )
			threshold = 0.5f;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = 2;

		return detectNet::Create(prototxt, modelName, meanPixel, class_labels, threshold, input, out_cvg, out_bbox, maxBatchSize);
	}

	// create segnet from pretrained model
	return detectNet::Create(type);
}
	

// Create (UFF)
detectNet* detectNet::Create( const char* model, const char* class_labels, float threshold, 
						const char* input, const Dims3& inputDims, 
						const char* output, const char* numDetections,
						uint32_t maxBatchSize, precisionType precision,
				   		deviceType device, bool allowGPUFallback )
{
	detectNet* net = new detectNet(0.0f);
	
	if( !net )
		return NULL;

	printf("\n");
	printf("detectNet -- loading detection network model from:\n");
	printf("          -- model        %s\n", CHECK_NULL_STR(model));
	printf("          -- input_blob   '%s'\n", CHECK_NULL_STR(input));
	printf("          -- output_blob  '%s'\n", CHECK_NULL_STR(output));
	printf("          -- output_count '%s'\n", CHECK_NULL_STR(numDetections));
	printf("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
	printf("          -- threshold    %f\n", threshold);
	printf("          -- batch_size   %u\n\n", maxBatchSize);
	
	//net->EnableDebug();
	
	// create list of output names	
	std::vector<std::string> output_blobs;

	if( output != NULL )
		output_blobs.push_back(output);

	if( numDetections != NULL )
		output_blobs.push_back(numDetections);
	
	// load the model
	if( !net->LoadNetwork(NULL, model, NULL, input, inputDims, output_blobs, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("detectNet -- failed to initialize.\n");
		return NULL;
	}
	
	// allocate detection sets           
	if( !net->allocDetections() )
		return NULL;

	// load class descriptions
	net->loadClassDesc(class_labels);
	net->defaultClassDesc();
	
	// set default class colors
	if( !net->defaultColors() )
		return NULL;

	// set the specified threshold
	net->SetThreshold(threshold);

	return net;
}

// Detect
int detectNet::Detect( float* input, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay )
{
	Detection* det = mDetectionSets[0] + mDetectionSet * GetMaxDetections();

	if( detections != NULL )
		*detections = det;

	mDetectionSet++;

	if( mDetectionSet >= mNumDetectionSets )
		mDetectionSet = 0;
	
	return Detect(input, width, height, det, overlay);
}

// Detect
int detectNet::Detect( float* rgba, uint32_t width, uint32_t height, Detection* detections, uint32_t overlay )
{
	if( !rgba || width == 0 || height == 0 || !detections )
	{
		printf(LOG_TRT "detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_UFF) )
	{
		if( CUDA_FAILED(cudaPreImageNetNormBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
										  make_float2(-1.0f, 1.0f), GetStream())) )
		{
			printf(LOG_TRT "detectNet::Detect() -- cudaPreImageNetNorm() failed\n");
			return -1;
		}
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
		if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, 
										   make_float2(0.0f, 1.0f), 
										   make_float3(0.485f, 0.456f, 0.406f),
										   make_float3(0.229f, 0.224f, 0.225f), 
										   GetStream())) )
		{
			printf(LOG_TRT "imageNet::PreProcess() -- cudaPreImageNetNormMeanRGB() failed\n");
			return false;
		}
	}
	else
	{
		if( mMeanPixel != 0.0f )
		{
			if( CUDA_FAILED(cudaPreImageNetMeanBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
										  make_float3(mMeanPixel, mMeanPixel, mMeanPixel), GetStream())) )
			{
				printf(LOG_TRT "detectNet::Detect() -- cudaPreImageNetMean() failed\n");
				return -1;
			}
		}
		else
		{
			if( CUDA_FAILED(cudaPreImageNetBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, GetStream())) )
			{
				printf(LOG_TRT "detectNet::Detect() -- cudaPreImageNet() failed\n");
				return -1;
			}
		}
	}
	
	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);

	// process with TensorRT
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA, mOutputs[1].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_TRT "detectNet::Detect() -- failed to execute TensorRT context\n");
		return -1;
	}
	
	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// post-processing / clustering
	int numDetections = 0;

	if( IsModelType(MODEL_UFF) )
	{		
		const int rawDetections = *(int*)mOutputs[OUTPUT_NUM].CPU;
		const int rawParameters = DIMS_W(mOutputs[OUTPUT_UFF].dims);

#ifdef DEBUG_CLUSTERING	
		printf(LOG_TRT "detectNet::Detect() -- %i unfiltered detections\n", rawDetections);
#endif

		// filter the raw detections by thresholding the confidence
		for( int n=0; n < rawDetections; n++ )
		{
			float* object_data = mOutputs[OUTPUT_UFF].CPU + n * rawParameters;

			if( object_data[2] < mCoverageThreshold )
				continue;

			detections[numDetections].Instance   = numDetections; //(uint32_t)object_data[0];
			detections[numDetections].ClassID    = (uint32_t)object_data[1];
			detections[numDetections].Confidence = object_data[2];
			detections[numDetections].Left       = object_data[3] * width;
			detections[numDetections].Top        = object_data[4] * height;
			detections[numDetections].Right      = object_data[5] * width;
			detections[numDetections].Bottom	  = object_data[6] * height;

			if( detections[numDetections].ClassID >= mNumClasses )
			{
				printf(LOG_TRT "detectNet::Detect() -- detected object has invalid classID (%u)\n", detections[numDetections].ClassID);
				detections[numDetections].ClassID = 0;
			}

			if( strcmp(GetClassDesc(detections[(uint32_t)numDetections].ClassID), "void") == 0 )
				continue;

			numDetections += clusterDetections(detections, numDetections);
		}

		// sort the detections by confidence value
		sortDetections(detections, numDetections);
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		float* coord = mOutputs[0].CPU;

		coord[0] = ((coord[0] + 1.0f) * 0.5f) * float(width);
		coord[1] = ((coord[1] + 1.0f) * 0.5f) * float(height);
		coord[2] = ((coord[2] + 1.0f) * 0.5f) * float(width);
		coord[3] = ((coord[3] + 1.0f) * 0.5f) * float(height);

		printf(LOG_TRT "detectNet::Detect() -- ONNX -- coord (%f, %f) (%f, %f)  image %ux%u\n", coord[0], coord[1], coord[2], coord[3], width, height);

		detections[numDetections].Instance   = numDetections;
		detections[numDetections].ClassID    = 0;
		detections[numDetections].Confidence = 1;
		detections[numDetections].Left       = coord[0];
		detections[numDetections].Top        = coord[1];
		detections[numDetections].Right      = coord[2];
		detections[numDetections].Bottom	  = coord[3];	
	
		numDetections++;
	}
	else
	{
		// cluster detections
		numDetections = clusterDetections(detections, width, height);
	}

	PROFILER_END(PROFILER_POSTPROCESS);

	// render the overlay
	if( overlay != 0 && numDetections > 0 )
	{
		if( !Overlay(rgba, rgba, width, height, detections, numDetections, overlay) )
			printf(LOG_TRT "detectNet::Detect() -- failed to render overlay\n");
	}
	
	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// return the number of detections
	return numDetections;
}

// clusterDetections (UFF)
int detectNet::clusterDetections( Detection* detections, int n, float threshold )
{
	if( n == 0 )
		return 1;

	// test each detection to see if it intersects
	for( int m=0; m < n; m++ )
	{
		if( detections[n].Intersects(detections[m], threshold) )	// TODO NMS or different threshold for same classes?
		{
			// if the intersecting detections have different classes, pick the one with highest confidence
			// otherwise if they have the same object class, expand the detection bounding box
			if( detections[n].ClassID != detections[m].ClassID )
			{
				if( detections[n].Confidence > detections[m].Confidence )
				{
					detections[m] = detections[n];

					detections[m].Instance = m;
					detections[m].ClassID = detections[n].ClassID;
					detections[m].Confidence = detections[n].Confidence;					
				}
			}
			else
			{
				detections[m].Expand(detections[n]);
				detections[m].Confidence = fmaxf(detections[n].Confidence, detections[m].Confidence);
			}

			return 0; // merged detection
		}
	}

	return 1;	// new detection
}

// sortDetections (UFF)
void detectNet::sortDetections( Detection* detections, int numDetections )
{
	if( numDetections < 2 )
		return;

	// order by area (descending) or confidence (ascending)
	for( int i=0; i < numDetections-1; i++ )
	{
		for( int j=0; j < numDetections-i-1; j++ )
		{
			if( detections[j].Area() < detections[j+1].Area() ) //if( detections[j].Confidence > detections[j+1].Confidence )
			{
				const Detection det = detections[j];
				detections[j] = detections[j+1];
				detections[j+1] = det;
			}
		}
	}

	// renumber the instance ID's
	for( int i=0; i < numDetections; i++ )
		detections[i].Instance = i;	
}

// from detectNet.cu
cudaError_t cudaDetectionOverlay( float4* input, float4* output, uint32_t width, uint32_t height, detectNet::Detection* detections, int numDetections, float4* colors );

// Overlay
bool detectNet::Overlay( float* input, float* output, uint32_t width, uint32_t height, Detection* detections, uint32_t numDetections, uint32_t flags )
{
	PROFILER_BEGIN(PROFILER_VISUALIZE);

	if( flags == 0 )
	{
		printf(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_NONE, returning false\n");
		return false;
	}

	// bounding box overlay
	if( flags & OVERLAY_BOX )
	{
		if( CUDA_FAILED(cudaDetectionOverlay((float4*)input, (float4*)output, width, height, detections, numDetections, (float4*)mClassColors[1])) )
			return false;
	}

	// class label overlay
	if( (flags & OVERLAY_LABEL) || (flags & OVERLAY_CONFIDENCE) )
	{
		static cudaFont* font = NULL;

		// make sure the font object is created
		if( !font )
		{
			font = cudaFont::Create(adaptFontSize(width));
	
			if( !font )
			{
				printf(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_FONT, but failed to create cudaFont()\n");
				return false;
			}
		}

		// draw each object's description
		std::vector< std::pair< std::string, int2 > > labels;

		for( uint32_t n=0; n < numDetections; n++ )
		{
			const char* className  = GetClassDesc(detections[n].ClassID);
			const float confidence = detections[n].Confidence * 100.0f;
			const int2  position   = make_int2(detections[n].Left+5, detections[n].Top+3);
			
			if( flags & OVERLAY_CONFIDENCE )
			{
				char str[256];

				if( (flags & OVERLAY_LABEL) && (flags & OVERLAY_CONFIDENCE) )
					sprintf(str, "%s %.1f%%", className, confidence);
				else
					sprintf(str, "%.1f%%", confidence);

				labels.push_back(std::pair<std::string, int2>(str, position));
			}
			else
			{
				// overlay label only
				labels.push_back(std::pair<std::string, int2>(className, position));
			}
		}

		font->OverlayText((float4*)input, width, height, labels, make_float4(255,255,255,255));
	}
	
	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// allocDetections
bool detectNet::allocDetections()
{
	// determine max detections
	if( IsModelType(MODEL_UFF) )	// TODO:  fixme
	{
		printf("W = %u  H = %u  C = %u\n", DIMS_W(mOutputs[OUTPUT_UFF].dims), DIMS_H(mOutputs[OUTPUT_UFF].dims), DIMS_C(mOutputs[OUTPUT_UFF].dims));
		mMaxDetections = DIMS_H(mOutputs[OUTPUT_UFF].dims) * DIMS_C(mOutputs[OUTPUT_UFF].dims);
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		mMaxDetections = 1;
		mNumClasses = 1;
		printf("detectNet -- using ONNX model\n");
	}	
	else
	{
		mNumClasses = DIMS_C(mOutputs[OUTPUT_CVG].dims);
		mMaxDetections = DIMS_W(mOutputs[OUTPUT_CVG].dims) * DIMS_H(mOutputs[OUTPUT_CVG].dims) /** DIMS_C(mOutputs[OUTPUT_CVG].dims)*/ * mNumClasses;
		printf("detectNet -- number object classes:   %u\n", mNumClasses);
	}

	printf("detectNet -- maximum bounding boxes:  %u\n", mMaxDetections);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * mNumDetectionSets * mMaxDetections;
	
	if( !cudaAllocMapped((void**)&mDetectionSets[0], (void**)&mDetectionSets[1], det_size) )
		return false;
	
	memset(mDetectionSets[0], 0, det_size);
	return true;
}

cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream );	
cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream );



struct float6 { float x; float y; float z; float w; float v; float u; };
static inline float6 make_float6( float x, float y, float z, float w, float v, float u ) { float6 f; f.x = x; f.y = y; f.z = z; f.w = w; f.v = v; f.u = u; return f; }


inline static bool rectOverlap(const float6& r1, const float6& r2)
{
    return ! ( r2.x > r1.z  
        || r2.z < r1.x
        || r2.y > r1.w
        || r2.w < r1.y
        );
}

static void mergeRect( std::vector<float6>& rects, const float6& rect )
{
	const uint32_t num_rects = rects.size();
	
	bool intersects = false;
	
	for( uint32_t r=0; r < num_rects; r++ )
	{
		if( rectOverlap(rects[r], rect) )
		{
			intersects = true;   

#ifdef DEBUG_CLUSTERING
			printf("found overlap\n");		
#endif

			if( rect.x < rects[r].x ) 	rects[r].x = rect.x;
			if( rect.y < rects[r].y ) 	rects[r].y = rect.y;
			if( rect.z > rects[r].z )	rects[r].z = rect.z;
			if( rect.w > rects[r].w ) 	rects[r].w = rect.w;
			
			break;
		}
			
	} 
	
	if( !intersects )
		rects.push_back(rect);
}


// Detect
bool detectNet::Detect( float* rgba, uint32_t width, uint32_t height, float* boundingBoxes, int* numBoxes, float* confidence )
{
	if( !rgba || width == 0 || height == 0 || !boundingBoxes || !numBoxes || *numBoxes < 1 )
	{
		printf("detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return false;
	}

	std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << std::endl;
	// downsample and convert to band-sequential BGR
	if( mMeanPixel != 0.0f )
	{
		if( CUDA_FAILED(cudaPreImageNetMean((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
									  make_float3(mMeanPixel, mMeanPixel, mMeanPixel), GetStream())) )
		{
			printf("detectNet::Classify() -- cudaPreImageNetMean failed\n");
			return false;
		}
	}
	else
	{
		if( CUDA_FAILED(cudaPreImageNet((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, GetStream())) )
		{
			printf("detectNet::Classify() -- cudaPreImageNet failed\n");
			return false;
		}
	}
	std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << std::endl;
	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[OUTPUT_CVG].CUDA, mOutputs[OUTPUT_BBOX].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_GIE "detectNet::Classify() -- failed to execute tensorRT context\n");
		*numBoxes = 0;
		return false;
	}
	std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << std::endl;
	PROFILER_REPORT();

	// cluster detection bboxes
	float* net_cvg   = mOutputs[OUTPUT_CVG].CPU;
	float* net_rects = mOutputs[OUTPUT_BBOX].CPU;
	
	const int ow  = DIMS_W(mOutputs[OUTPUT_BBOX].dims);		// number of columns in bbox grid in X dimension
	const int oh  = DIMS_H(mOutputs[OUTPUT_BBOX].dims);		// number of rows in bbox grid in Y dimension
	const int owh = ow * oh;							// total number of bbox in grid
	const int cls = GetNumClasses();					// number of object classes in coverage map
	
	const float cell_width  = /*width*/ DIMS_W(mInputDims) / ow;
	const float cell_height = /*height*/ DIMS_H(mInputDims) / oh;
	
	const float scale_x = float(width) / float(DIMS_W(mInputDims));
	const float scale_y = float(height) / float(DIMS_H(mInputDims));

#ifdef DEBUG_CLUSTERING	
	printf("input width %i height %i\n", (int)DIMS_W(mInputDims), (int)DIMS_H(mInputDims));
	printf("cells x %i  y %i\n", ow, oh);
	printf("cell width %f  height %f\n", cell_width, cell_height);
	printf("scale x %f  y %f\n", scale_x, scale_y);
#endif
#if 1
	std::vector< std::vector<float6> > rects;
	rects.resize(cls);
	std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << std::endl;
	// extract and cluster the raw bounding boxes that meet the coverage threshold
	for( uint32_t z=0; z < cls; z++ )
	{
		rects[z].reserve(owh);
		
		for( uint32_t y=0; y < oh; y++ )
		{
			for( uint32_t x=0; x < ow; x++)
			{
				const float coverage = net_cvg[z * owh + y * ow + x];
				
				if( coverage > mCoverageThreshold )
				{
					const float mx = x * cell_width;
					const float my = y * cell_height;
					
					const float x1 = (net_rects[0 * owh + y * ow + x] + mx) * scale_x;	// left
					const float y1 = (net_rects[1 * owh + y * ow + x] + my) * scale_y;	// top
					const float x2 = (net_rects[2 * owh + y * ow + x] + mx) * scale_x;	// right
					const float y2 = (net_rects[3 * owh + y * ow + x] + my) * scale_y;	// bottom 
					
				#ifdef DEBUG_CLUSTERING
					printf("rect x=%u y=%u  cvg=%f  %f %f   %f %f \n", x, y, coverage, x1, x2, y1, y2);
				#endif					
					mergeRect( rects[z], make_float6(x1, y1, x2, y2, coverage, z) );
				}
			}
		}
	}
	
	//printf("done clustering rects\n");
	std::cout << __PRETTY_FUNCTION__ << " " << __LINE__ << std::endl;
	// condense the multiple class lists down to 1 list of detections
	const uint32_t numMax = *numBoxes;
	int n = 0;
	
	for( uint32_t z = 0; z < cls; z++ )
	{
		const uint32_t numBox = rects[z].size();
		
		for( uint32_t b = 0; b < numBox && n < numMax; b++ )
		{
			const float6 r = rects[z][b];
			
			boundingBoxes[n * 4 + 0] = r.x;
			boundingBoxes[n * 4 + 1] = r.y;
			boundingBoxes[n * 4 + 2] = r.z;
			boundingBoxes[n * 4 + 3] = r.w;
			
			if( confidence != NULL )
			{
				confidence[n * 2 + 0] = r.v;	// coverage
				confidence[n * 2 + 1] = r.u;	// class ID
			}
			
			n++;
		}
	}
	
	*numBoxes = n;
#else
	*numBoxes = 0;
#endif
	return true;
}


// DrawBoxes
bool detectNet::DrawBoxes( float* input, float* output, uint32_t width, uint32_t height, const float* boundingBoxes, int numBoxes, int classIndex )
{
	if( !input || !output || width == 0 || height == 0 || !boundingBoxes || numBoxes < 1 || classIndex < 0 || classIndex >= GetNumClasses() )
		return false;
	
	const float4 color = make_float4( mClassColors[0][classIndex*4+0], 
									  mClassColors[0][classIndex*4+1],
									  mClassColors[0][classIndex*4+2],
									  mClassColors[0][classIndex*4+3] );
	
	//printf("draw boxes  %i  %i   %f %f %f %f\n", numBoxes, classIndex, color.x, color.y, color.z, color.w);
	
	if( CUDA_FAILED(cudaRectOutlineOverlay((float4*)input, (float4*)output, width, height, (float4*)boundingBoxes, numBoxes, color)) )
		return false;
	
	return true;
}
	

// SetClassColor
void detectNet::SetClassColor( uint32_t classIndex, float r, float g, float b, float a )
{
	if( classIndex >= GetNumClasses() || !mClassColors[0] )
		return;
	
	const uint32_t i = classIndex * 4;
	
	mClassColors[0][i+0] = r;
	mClassColors[0][i+1] = g;
	mClassColors[0][i+2] = b;
	mClassColors[0][i+3] = a;
}
