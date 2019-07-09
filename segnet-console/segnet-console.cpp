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

#include "segNet.h"
#include <deque>
#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"
#include <iostream>
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include <algorithm>

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

std::deque<std::string> readDir(const char * imgDir){
DIR *dir;
struct dirent *ent;
std::deque<std::string> files;
if ((dir = opendir (imgDir)) != NULL) {
  /* print all the files and directories within directory */
  while ((ent = readdir (dir)) != NULL) {
	  std::string f_name(ent->d_name);
	  if (f_name.find(".png") != std::string::npos && f_name.find(".json") == std::string::npos){
    		files.push_back( f_name);
	  }
  }
  closedir (dir);
} else {
  /* could not open directory */
  perror ("");
  return files;
}
std::sort(files.begin(), files.end());

return files;
}

// main entry point
int main( int argc, char** argv )
{
	printf("segnet-console\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	

        // load image from file on disk
        float* imgCPU    = NULL;
        float* imgCUDA   = NULL;
        int    imgWidth  = 0;
        int    imgHeight = 0;
        float* outCPU  = NULL;
        float* outCUDA = NULL;


#ifdef __x86_64__
        segNet* net = segNet::Create("FCN-Alexnet-Cityscapes-HD/deploy.prototxt", "FCN-Alexnet-Cityscapes-HD/snapshot_iter_367568.caffemodel", "FCN-Alexnet-Cityscapes-HD/cityscapes-labels.txt", "FCN-Alexnet-Cityscapes-HD/cityscapes-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, 4, TYPE_FP32);
#else
        segNet* net = segNet::Create("FCN-Alexnet-Cityscapes-HD/deploy.prototxt", "FCN-Alexnet-Cityscapes-HD/snapshot_iter_367568.caffemodel", "FCN-Alexnet-Cityscapes-HD/cityscapes-labels.txt", "FCN-Alexnet-Cityscapes-HD/cityscapes-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, 4, TYPE_INT8);
#endif
        //net->SetGlobalAlpha(120);
        net->EnableProfiler();

	const char * imageDirectory = argv[1];
	std::string imgDirString(imageDirectory);
	std::string imageOutputDirectory = argv[2];
	auto files = readDir(imageDirectory);
	for (auto & f : files){
		std::string img_str = imgDirString + f; 
		imgWidth = 2048 / 2;
		imgHeight = (1536) / 2;
        	if( !loadImageRGBA(img_str.c_str(), (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
        	{
                	printf("failed to load image '%s'\n", img_str.c_str());
                	return 0;
        	}
		if (outCUDA == NULL && outCPU == NULL){
        		if( !cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, imgWidth * imgHeight * sizeof(float) * 4) )
        		{
                		printf("segnet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
                		return 0;
        		}
		}

        	// process the segmentation network
	        printf("segnet-console:  beginning processing (%zu)\n", current_timestamp());

        	if( !net->Process(imgCUDA, imgWidth, imgHeight) )
        	{
                	printf("segnet-console:  failed to process segmentation\n");
                	return 0;
        	}
	        CUDA(cudaThreadSynchronize());
		// generate image overlay
	        if( !net->Overlay(outCUDA, imgWidth, imgHeight, segNet::FILTER_LINEAR) )
        	{
                	printf("segnet-console:  failed to generate overlay.\n");
                	return 0;
        	}
		std::string outFilename = imageOutputDirectory + "/" + f; 
		if( !saveImageRGBA(outFilename.c_str(), (float4*)outCPU, imgWidth, imgHeight) )
		{
			printf("ERROR WRITING TO DISK\n");
		}

	}
	
/*	
	// retrieve filename arguments
	if( argc < 2 )
	{
		printf("segnet-console:   input image filename required\n");
		return 0;
	}

	if( argc < 3 )
	{
		printf("segnet-console:   output image filename required\n");
		return 0;
	}
	
	const char* imgFilename = argv[1];
	const char* outFilename = argv[2];


	// create the segNet from pretrained or custom model by parsing the command line
	segNet* net = segNet::Create(argc, argv);

	if( !net )
	{
		printf("segnet-console:   failed to initialize segnet\n");
		return 0;
	}
	
	// enable layer timings for the console application
	net->EnableProfiler();

	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}

	// allocate output image
	float* outCPU  = NULL;
	float* outCUDA = NULL;

	if( !cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, imgWidth * imgHeight * sizeof(float) * 4) )
	{
		printf("segnet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
		return 0;
	}

	// set alpha blending value for classes that don't explicitly already have an alpha	
	net->SetGlobalAlpha(120);


	// process the segmentation network
	printf("segnet-console:  beginning processing (%zu)\n", current_timestamp());

	if( !net->Process(imgCUDA, imgWidth, imgHeight) )
	{
		printf("segnet-console:  failed to process segmentation\n");
		return 0;
	}

	printf("segnet-console:  finished processing (%zu)\n", current_timestamp());


	// generate image overlay
	if( !net->Overlay(outCUDA, imgWidth, imgHeight, segNet::FILTER_LINEAR) )
	{
		printf("segnet-console:  failed to generate overlay.\n");
		return 0;
	}

	CUDA(cudaThreadSynchronize());


	// save output image
	if( !saveImageRGBA(outFilename, (float4*)outCPU, imgWidth, imgHeight) )
		printf("segnet-console:  failed to save output image to '%s'\n", outFilename);
	else
		printf("segnet-console:  completed saving '%s'\n", outFilename);

	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	CUDA(cudaFreeHost(outCPU));
	delete net;
	*/
	return 0;
}
