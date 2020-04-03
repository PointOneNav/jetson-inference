/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "superResNet.h"
#include <fstream>
#include <iostream>
#include "cudaUtility.h"

// constructor
superResNet::superResNet() {}

// Destructor
superResNet::~superResNet() {}
// Create
superResNet* superResNet::Create(NetworkType networkType, uint32_t maxBatchSize) {
  superResNet* net = NULL;
  /*
  FCN_ResNet18_Cityscapes_512x256,
    FCN_ResNet18_Cityscapes_1024x512,
    FCN_ResNet18_Cityscapes_2048x1024,
    FCN_ResNet18_DeepScene_576x320,
    FCN_ResNet18_DeepScene_864x480,
    FCN_ResNet18_MHP_512x320,
    FCN_ResNet18_MHP_640x360,
    FCN_ResNet18_Pascal_VOC_320x320,
    FCN_ResNet18_Pascal_VOC_512x320,
    FCN_ResNet18_SUN_RGBD_512x400,
    FCN_ResNet18_SUN_RGBD_640x512*/
  if(networkType == FCN_ResNet18_Cityscapes_2048x1024) {
    net = Create("third_party/jetson-inference/data/FCN-ResNet18-Cityscapes-2048x1024/"
                  "fcn_resnet18.onnx",
                  RESNET_DEFAULT_INPUT, RESNET_DEFAULT_OUTPUT, maxBatchSize);
  } 
  return net;
}

superResNet* superResNet::Create(const char* model, 
                                  const char* input_blob, const char* output_blob,
                                  uint32_t maxBatchSize) { 
  superResNet* net = new superResNet(); 

  if (!net->LoadNetwork(NULL, model, NULL, input_blob, output_blob,
                        maxBatchSize, TYPE_FP32)) {
    printf(LOG_TRT "failed to load superResNet model\n");
    return NULL;
  }

  printf("\n");
  printf("superResNet -- super resolution network loaded from:\n");
  printf("            -- model        '%s'\n", model);
  printf("            -- input blob   '%s'\n", input_blob);
  printf("            -- output blob  '%s'\n", output_blob);
  printf("            -- batch size   %u\n", maxBatchSize);
  printf("            -- input dims   %ux%u\n", net->GetInputWidth(),
         net->GetInputHeight());
  printf("            -- output dims  %ux%u\n", net->GetOutputWidth(),
         net->GetOutputHeight());
  printf("            -- scale factor %.8fx\n\n", net->GetScaleFactor());

  return net;
}

// Create
superResNet* superResNet::Create() {
#ifndef HAS_SUPERRES_NET
  printf(LOG_TRT
         "error -- superResNet is supported only in TensorRT 5.0 and newer\n");
  return NULL;
#endif

  superResNet* net = new superResNet();

  //   const char* model_path =
  //       "networks/Super-Resolution-BSD500/super_resolution_bsd500.onnx";
  const char* model_path =
      "third_party/jetson-inference/data/FCN-ResNet18-Cityscapes-2048x1024/"
      "fcn_resnet18.onnx";
  std::ifstream ifff(model_path);
  const char* input_blob = "input_0";
  const char* output_blob = "output_0";

  const uint32_t maxBatchSize = 1;

  if (!net->LoadNetwork(NULL, model_path, NULL, input_blob, output_blob,
                        maxBatchSize, TYPE_FP32)) {
    printf(LOG_TRT "failed to load superResNet model\n");
    return NULL;
  }

  printf("\n");
  printf("superResNet -- super resolution network loaded from:\n");
  printf("            -- model        '%s'\n", model_path);
  printf("            -- input blob   '%s'\n", input_blob);
  printf("            -- output blob  '%s'\n", output_blob);
  printf("            -- batch size   %u\n", maxBatchSize);
  printf("            -- input dims   %ux%u\n", net->GetInputWidth(),
         net->GetInputHeight());
  printf("            -- output dims  %ux%u\n", net->GetOutputWidth(),
         net->GetOutputHeight());
  printf("            -- scale factor %.8fx\n\n", net->GetScaleFactor());

  return net;
}

// cudaPreSuperResNet
cudaError_t cudaPreSuperResNet(float4* input, size_t inputWidth,
                               size_t inputHeight, float* output,
                               size_t outputWidth, size_t outputHeight,
                               float maxPixelValue, cudaStream_t stream);

// cudaPostSuperResNet
cudaError_t cudaPostSuperResNet(float* input, size_t inputWidth,
                                size_t inputHeight, float4* output,
                                size_t outputWidth, size_t outputHeight,
                                float maxPixelValue, cudaStream_t stream);

// UpscaleRGBA
bool superResNet::UpscaleRGBA(float* input, uint32_t inputWidth,
                              uint32_t inputHeight, float* output,
                              uint32_t outputWidth, uint32_t outputHeight,
                              float maxPixelValue) {
  /*
	 * convert input image to NCHW format and with pixel range 0.0-1.0f
	 */
  if (CUDA_FAILED(cudaPreSuperResNet(
          (float4*)input, inputWidth, inputHeight, mInputCUDA, GetInputWidth(),
          GetInputHeight(), maxPixelValue, GetStream()))) {
    printf(LOG_TRT
           "superResNet::UpscaleRGBA() -- cudaPreSuperResNet() failed\n");
    return false;
  }

  /*
	 * perform the inferencing
 	 */
  void* bindBuffers[] = {mInputCUDA, mOutputs[0].CUDA};

  if (!mContext->execute(1, bindBuffers)) {
    printf(
        LOG_TRT
        "superResNet::UpscaleRGBA() -- failed to execute TensorRT network\n");
    return false;
  }

  PROFILER_REPORT();

  /*
	 * convert output image from NCHW to packed RGBA, with the user's pixel range
	 */
  if (CUDA_FAILED(cudaPostSuperResNet(mOutputs[0].CUDA, GetOutputWidth(),
                                      GetOutputHeight(), (float4*)output,
                                      outputWidth, outputHeight, maxPixelValue,
                                      GetStream()))) {
    printf(LOG_TRT
           "superResNet::UpscaleRGBA() -- cudaPostSuperResNet() failed\n");
    return false;
  }

  return true;
}

// UpscaleRGBA
bool superResNet::UpscaleRGBA(float* input, float* output,
                              float maxPixelValue) {
  return UpscaleRGBA(input, GetInputWidth(), GetInputHeight(), output,
                     GetOutputWidth(), GetOutputHeight(), maxPixelValue);
}
