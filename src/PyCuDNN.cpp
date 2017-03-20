#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "PyCuDNNActivationDescriptor.hpp"
#include "PyCuDNNActivationMode.hpp"
#include "PyCuDNNBatchNormMode.hpp"
#include "PyCuDNNConvolutionBwdDataAlgo.hpp"
#include "PyCuDNNConvolutionBwdDataAlgoPerf.hpp"
#include "PyCuDNNConvolutionBwdDataPreference.hpp"
#include "PyCuDNNConvolutionBwdFilterAlgo.hpp"
#include "PyCuDNNConvolutionBwdFilterAlgoPerf.hpp"
#include "PyCuDNNConvolutionBwdFilterPreference.hpp"
#include "PyCuDNNConvolutionDescriptor.hpp"
#include "PyCuDNNConvolutionFwdAlgo.hpp"
#include "PyCuDNNConvolutionFwdAlgoPerf.hpp"
#include "PyCuDNNConvolutionFwdPreference.hpp"
#include "PyCuDNNConvolutionMode.hpp"
#include "PyCuDNNDataType.hpp"
#include "PyCuDNNDirectionMode.hpp"
#include "PyCuDNNDivNormMode.hpp"
#include "PyCuDNNDropoutDescriptor.hpp"
#include "PyCuDNNFilterDescriptor.hpp"
#include "PyCuDNNHandle.hpp"
#include "PyCuDNNLRNDescriptor.hpp"
#include "PyCuDNNLRNMode.hpp"
#include "PyCuDNNNanPropagation.hpp"
#include "PyCuDNNOpTensorDescriptor.hpp"
#include "PyCuDNNOpTensorOp.hpp"
#include "PyCuDNNPoolingDescriptor.hpp"
#include "PyCuDNNPoolingMode.hpp"
#include "PyCuDNNRNNDescriptor.hpp"
#include "PyCuDNNRNNInputMode.hpp"
#include "PyCuDNNRNNMode.hpp"
#include "PyCuDNNSamplerType.hpp"
#include "PyCuDNNSoftmaxAlgorithm.hpp"
#include "PyCuDNNSoftmaxMode.hpp"
#include "PyCuDNNSpatialTransformerDescriptor.hpp"
#include "PyCuDNNStatus.hpp"
#include "PyCuDNNTensorDescriptor.hpp"
#include "PyCuDNNTensorFormat.hpp"

namespace py = pybind11;

////////////////////////////////////////////////////////////////////////////////////////////////////
//  PyCuDNN API
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace PyCuDNN {

  auto getVersion ()
  {
    return cudnnGetVersion();
  }

  auto getErrorString ( const Status& status )
  {
    return cudnnGetErrorString(status);
  }

  auto setTensor4dDescriptor (
    TensorDescriptor& tensorDesc,
    TensorFormat tensorFormat,
    DataType dataType,
    std::tuple<int, int, int, int> tensorDims)
  {
    int n = std::get<0>(tensorDims);
    int c = std::get<1>(tensorDims);
    int h = std::get<2>(tensorDims);
    int w = std::get<3>(tensorDims);
    checkStatus(
      cudnnSetTensor4dDescriptor(tensorDesc, tensorFormat, dataType, n, c, h, w)
    );
  }

  auto setTensor4dDescriptorEx (
    TensorDescriptor& tensorDesc,
    DataType dataType,
    std::tuple<int, int, int, int> tensorDims,
    std::tuple<int, int, int, int> tensorStrides )
  {
    int n = std::get<0>(tensorDims);
    int c = std::get<1>(tensorDims);
    int h = std::get<2>(tensorDims);
    int w = std::get<3>(tensorDims);

    int nS = std::get<0>(tensorStrides);
    int cS = std::get<1>(tensorStrides);
    int hS = std::get<2>(tensorStrides);
    int wS = std::get<3>(tensorStrides);

    checkStatus(
      cudnnSetTensor4dDescriptorEx(tensorDesc, dataType, n, c, h, w, nS, cS, hS, wS)
    );
  }

  auto getTensor4dDescriptor ( TensorDescriptor& tensorDesc )
  {
    DataType dataType;
    std::vector<int> dims({0, 0, 0, 0});
    std::vector<int> strides({0, 0, 0, 0});

    checkStatus(
      cudnnGetTensor4dDescriptor(
        tensorDesc,
        &dataType,
        &dims[0],
        &dims[1],
        &dims[2],
        &dims[3],
        &strides[0],
        &strides[1],
        &strides[2],
        &strides[3]
      )
    );

    return std::make_tuple(dataType, dims, strides);
  }

  auto setTensorNdDescriptor (
    TensorDescriptor& tensorDesc,
    DataType dataType,
    std::vector<int> tensorDims,
    std::vector<int> tensorStrides )
  {
    if (tensorDims.size() != tensorStrides.size())
      throw std::length_error("tensorDims and tensorStrides must be of the same length");

    checkStatus(
      cudnnSetTensorNdDescriptor(
        tensorDesc,
        dataType,
        tensorDims.size(),
        tensorDims.data(),
        tensorStrides.data()
      )
    );
  }

  auto getTensorNdDescriptor ( TensorDescriptor& tensorDesc)
  {
    DataType dataType;

    int numTensorDims;
    std::vector<int> tensorDims;
    std::vector<int> tensorStrides;

    const size_t maxTensorDims = 16;
    int tensorDimsA[maxTensorDims];
    int tensorStridesA[maxTensorDims];

    checkStatus(
      cudnnGetTensorNdDescriptor(
        tensorDesc,
        maxTensorDims,
        &dataType,
        &numTensorDims,
        tensorDimsA,
        tensorStridesA
      )
    );

    for (int i = 0; i < numTensorDims; i++) {
      tensorDims.push_back(tensorDimsA[i]);
      tensorStrides.push_back(tensorStridesA[i]);
    }

    return std::make_tuple(dataType, tensorDims, tensorStrides);
  }

  auto transformTensor (
    const Handle& handle,
    double alpha,
    const TensorDescriptor& xDesc,
    const void* x,
    double beta,
    const TensorDescriptor& yDesc,
    void* y )
  {
    checkStatus(
      cudnnTransformTensor(handle, &alpha, xDesc, x, &beta, yDesc, y)
    );
  }

  auto addTensor (
    const Handle& handle,
    double alpha,
    const TensorDescriptor& xDesc,
    const void* x,
    double beta,
    const TensorDescriptor& yDesc,
    void* y )
  {
    checkStatus(
      cudnnAddTensor(handle, &alpha, xDesc, x, &beta, yDesc, y)
    );
  }

  auto opTensor (
    const Handle& handle,
    const OpTensorDescriptor& opTensorDesc,
    double alphaOne,
    const TensorDescriptor& aDesc,
    const void* A,
    double alphaTwo,
    const TensorDescriptor& bDesc,
    const void* B,
    double beta,
    const TensorDescriptor& cDesc,
    void* C )
  {
    checkStatus(
      cudnnOpTensor(handle, opTensorDesc, &alphaOne, aDesc, A, &alphaTwo, bDesc, B, &beta, cDesc, C)
    );
  }

  auto setTensor (
    const Handle& handle,
    const TensorDescriptor& yDesc,
    void* y,
    int value )
  {
    checkStatus(
      cudnnSetTensor(handle, yDesc, y, &value)
    );
  }

  auto scaleTensor (
    const Handle& handle,
    const TensorDescriptor& yDesc,
    void *y,
    double alpha )
  {
    checkStatus(
      cudnnScaleTensor(handle, yDesc, y, &alpha)
    );
  }

  void setFilter4dDescriptor (
    FilterDescriptor& filterDesc,
    const DataType& dataType,
    const TensorFormat& tensorFormat,
    std::tuple<int,int,int,int> tensorKCHW )
  {
    checkStatus(
      cudnnSetFilter4dDescriptor(
        filterDesc,
        dataType,
        tensorFormat,
        std::get<0>(tensorKCHW),
        std::get<1>(tensorKCHW),
        std::get<2>(tensorKCHW),
        std::get<3>(tensorKCHW)
      )
    );
  }

  auto getFilter4dDescriptor ( FilterDescriptor& filterDesc ) {
    DataType dataType;
    TensorFormat tensorFormat;
    int K, C, H, W;

    checkStatus(
      cudnnGetFilter4dDescriptor(
        filterDesc,
        &dataType,
        &tensorFormat,
        &K,
        &C,
        &H,
        &W
      )
    );

    return std::make_tuple(dataType, tensorFormat, std::vector<int>({K, C, H, W}));
  }

  void setFilterNdDescriptor (
    FilterDescriptor& filterDesc,
    const DataType& dataType,
    const TensorFormat& tensorFormat,
    std::vector<int> filterDims )
  {
    checkStatus(
      cudnnSetFilterNdDescriptor(
        filterDesc,
        dataType,
        tensorFormat,
        filterDims.size(),
        filterDims.data()
      )
    );
  }

  auto getFilterNdDescriptor( FilterDescriptor& filterDesc ) {
    DataType dataType;
    TensorFormat tensorFormat;

    int numFilterDims = 0;
    std::vector<int> filterDims;

    const size_t maxFilterDims = 16;
    int filterDimsA[maxFilterDims];

    checkStatus(
      cudnnGetFilterNdDescriptor(
        filterDesc,
        maxFilterDims,
        &dataType,
        &tensorFormat,
        &numFilterDims,
        filterDimsA
      )
    );

    for (int i = 0; i < numFilterDims; i++) {
      filterDims.push_back(filterDimsA[i]);
    }

    return std::make_tuple(dataType, tensorFormat, filterDims);
  }

  auto setConvolution2dDescriptor (
    const ConvolutionDescriptor& convDesc,
    int padH,
    int padW,
    int u,
    int v,
    int upscaleX,
    int upscaleY,
    const ConvolutionMode& mode )
  {

    checkStatus(
      cudnnSetConvolution2dDescriptor(
        convDesc.get(),
        padH,
        padW,
        u, v,
        upscaleX,
        upscaleY,
        mode
      )
    );
  }

  auto getConvolution2dDescriptor( ConvolutionDescriptor& convDesc ) {
    std::tuple<int,int,int,int,int,int,ConvolutionMode> result;

    checkStatus(
      cudnnSetConvolution2dDescriptor(
        convDesc,
        std::get<0>(result),
        std::get<1>(result),
        std::get<2>(result),
        std::get<3>(result),
        std::get<4>(result),
        std::get<5>(result),
        std::get<6>(result)
      )
    );

    return result;
  }

  auto getConvolution2dForwardOutputDim(
    ConvolutionDescriptor& convDesc,
    TensorDescriptor& inputTensorDesc,
    FilterDescriptor& filterDesc)
  {

    std::tuple<int,int,int,int> result;

    checkStatus(
      cudnnGetConvolution2dForwardOutputDim(
        convDesc,
        inputTensorDesc,
        filterDesc,
        &std::get<0>(result),
        &std::get<1>(result),
        &std::get<2>(result),
        &std::get<3>(result)
      )
    );

    return result;
  }

  void setConvolutionNdDescriptor(
    ConvolutionDescriptor& convDesc,
    std::vector<int> convPads,
    std::vector<int> filtStrides,
    std::vector<int> convUpscales,
    ConvolutionMode& convMode,
    DataType& dataType )
  {

    if ((convPads.size() != filtStrides.size()) ||
        (convPads.size() != convUpscales.size())) {
      throw std::length_error("convPads and filtStrides and convUpscales must be of the same length");
    }

    checkStatus(
      cudnnSetConvolutionNdDescriptor(
        convDesc,
        convPads.size(),
        convPads.data(),
        filtStrides.data(),
        convUpscales.data(),
        convMode,
        dataType
      )
    );
  }

  auto getConvolutionNdDescriptor ( ConvolutionDescriptor& convDesc ) {
    const int maxDims = 16;
    int padsArray[maxDims];
    int stridesArray[maxDims];
    int upscalesArray[maxDims];

    int numDims = 0;
    std::vector<int> pads;
    std::vector<int> strides;
    std::vector<int> upscales;

    ConvolutionMode mode;
    DataType dataType;

    checkStatus(
      cudnnGetConvolutionNdDescriptor(
        convDesc,
        maxDims,
        &numDims,
        padsArray,
        stridesArray,
        upscalesArray,
        &mode,
        &dataType
      )
    );

    for (int i = 0; i < numDims; i++) {
      pads.push_back(padsArray[i]);
      strides.push_back(stridesArray[i]);
      upscales.push_back(upscalesArray[i]);
    }

    return std::make_tuple(pads, strides, upscales, mode, dataType);
  }

  auto getConvolutionNdForwardOutputDim (
    ConvolutionDescriptor& convDesc,
    TensorDescriptor& inputTensorDesc,
    FilterDescriptor& filterDesc,
    int numDims )
  {

    std::vector<int> tensorOuputDims;
    for (int i = 0; i < numDims; i++)
      tensorOuputDims.push_back(0);

    checkStatus(
      cudnnGetConvolutionNdForwardOutputDim(
        convDesc,
        inputTensorDesc,
        filterDesc,
        numDims,
        tensorOuputDims.data()
      )
    );

    return tensorOuputDims;
  }

  auto findConvolutionForwardAlgorithm (
    Handle& handle,
    TensorDescriptor& xDesc,
    FilterDescriptor& wDesc,
    ConvolutionDescriptor& convDesc,
    TensorDescriptor& yDesc,
    int requestedAlgoCount = 32 )
  {
    const int maxAlgoCount = 32;
    if (requestedAlgoCount > maxAlgoCount) {
      throw std::range_error("exceeding max algorithm count (32)");
    }

    int returnedAlgoCount;
    ConvolutionFwdAlgoPerf perfResultsArray[32];

    checkStatus(
      cudnnFindConvolutionForwardAlgorithm(
        handle,
        xDesc,
        wDesc,
        convDesc,
        yDesc,
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResultsArray
      )
    );

    std::vector<ConvolutionFwdAlgoPerf> perfResults;
    for (int i = 0; i < returnedAlgoCount; i++) {
      perfResults.push_back(perfResultsArray[i]);
    }

    return perfResults;
  }

  auto findConvolutionForwardAlgorithmEx (
    Handle& handle,
    TensorDescriptor& xDesc,
    const void* x,
    FilterDescriptor& wDesc,
    const void* w,
    ConvolutionDescriptor convDesc,
    TensorDescriptor& yDesc,
    void *y,
    const int requestedAlgoCount,
    void* workSpace,
    size_t workSpaceSizeInBytes)
  {

    const int maxAlgoCount = 32;
    if (requestedAlgoCount > maxAlgoCount) {
      throw std::range_error("exceeding max algorithm count (32)");
    }

    int returnedAlgoCount;
    ConvolutionFwdAlgoPerf perfResultsArray[32];

    checkStatus(
      cudnnFindConvolutionForwardAlgorithmEx(
        handle,
        xDesc,
        x,
        wDesc,
        w,
        convDesc,
        yDesc,
        y,
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResultsArray,
        workSpace,
        workSpaceSizeInBytes
      )
    );

    std::vector<ConvolutionFwdAlgoPerf> perfResults;
    for (int i = 0; i < returnedAlgoCount; i++) {
      perfResults.push_back(perfResultsArray[i]);
    }

    return perfResults;
  }

  auto getConvolutionForwardAlgorithm (
    Handle& handle,
    TensorDescriptor& xDesc,
    FilterDescriptor& wDesc,
    ConvolutionDescriptor& convDesc,
    TensorDescriptor& yDesc,
    ConvolutionFwdPreference& preference,
    size_t memoryLimitInbytes )
  {
    ConvolutionFwdAlgo result;

    checkStatus(
      cudnnGetConvolutionForwardAlgorithm(
        handle,
        xDesc,
        wDesc,
        convDesc,
        yDesc,
        preference,
        memoryLimitInbytes,
        &result)
    );

    return result;
  }

  auto convolutionForward (
    Handle&                                   handle,
    double                                    alpha,
    TensorDescriptor&                         xDesc,
    const void*                               x,
    FilterDescriptor&                         wDesc,
    const void*                               w,
    ConvolutionDescriptor&                    convDesc,
    ConvolutionFwdAlgo&                       algo,
    void*                                     workSpace,
    size_t                                    workSpaceSizeInBytes,
    double                                    beta,
    TensorDescriptor&                         yDesc,
    void* y )
  {
    checkStatus(
      cudnnConvolutionForward(
        handle,
        &alpha,
        xDesc,
        x,
        wDesc,
        w,
        convDesc,
        algo,
        workSpace,
        workSpaceSizeInBytes,
        &beta,
        yDesc,
        y)
    );
  }

  void convolutionBackwardBias (
    Handle&                                   handle,
    double                                    alpha,
    TensorDescriptor&                         dyDesc,
    const void*                               dy,
    double                                    beta,
    TensorDescriptor&                         dbDesc,
    void                                      *db )
  {
    checkStatus(
      cudnnConvolutionBackwardBias(
        handle,
        &alpha,
        dyDesc,
        dy,
        &beta,
        dbDesc,
        db
      )
    );
  }

  auto findConvolutionBackwardFilterAlgorithm (
    const Handle& handle,
    const TensorDescriptor& xDesc,
    const TensorDescriptor& dyDesc,
    const ConvolutionDescriptor& convDesc,
    const FilterDescriptor& dwDesc,
    const int requestedAlgoCount = 32 )
  {
    const int maxAlgoCount = 32;
    if (requestedAlgoCount > maxAlgoCount) {
      throw std::range_error("exceeding max algorithm count (32)");
    }

    int returnedAlgoCount;
    ConvolutionBwdFilterAlgoPerf perfResultsArray[32];

    checkStatus(
      cudnnFindConvolutionBackwardFilterAlgorithm(
        handle,
        xDesc,
        dyDesc,
        convDesc,
        dwDesc,
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResultsArray
      )
    );

    std::vector<ConvolutionBwdFilterAlgoPerf> perfResults;
    for (int i = 0; i < returnedAlgoCount; i++) {
      perfResults.push_back(perfResultsArray[i]);
    }

    return perfResults;
  }

  auto findConvolutionBackwardFilterAlgorithmEx (
    const Handle&                             handle,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const TensorDescriptor&                   dyDesc,
    const void                                *y,
    const ConvolutionDescriptor&              convDesc,
    const FilterDescriptor&                   dwDesc,
    void                                      *dw,
    const int                                 requestedAlgoCount,
    void                                      *workSpace,
    size_t                                    workSpaceSizeInBytes )
  {
    const int maxAlgoCount = 32;
    if (requestedAlgoCount > maxAlgoCount) {
      throw std::range_error("exceeding max algorithm count (32)");
    }

    int returnedAlgoCount;
    ConvolutionBwdFilterAlgoPerf perfResultsArray[32];

    checkStatus(
      cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle,
        xDesc,
        x,
        dyDesc,
        y,
        convDesc,
        dwDesc,
        dw,
        requestedAlgoCount,
        &returnedAlgoCount,
        perfResultsArray,
        workSpace,
        workSpaceSizeInBytes
      )
    );

    std::vector<ConvolutionBwdFilterAlgoPerf> perfResults;
    for (int i = 0; i < returnedAlgoCount; i++) {
      perfResults.push_back(perfResultsArray[i]);
    }

    return perfResults;
  }

  auto getConvolutionBackwardFilterWorkspaceSize (
    const Handle&                             handle,
    const TensorDescriptor&                   xDesc,
    const TensorDescriptor&                   dyDesc,
    const ConvolutionDescriptor&              convDesc,
    const FilterDescriptor                    gradDesc,
    ConvolutionBwdFilterAlgo                  algo,
    size_t                                    *sizeInBytes )
  {

    // TODO: implement this
  }

  auto convolutionBackwardFilter(
    const Handle&                             handle,
    const void                                *alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const ConvolutionDescriptor&              convDesc,
    ConvolutionBwdFilterAlgo                  algo,
    void                                      *workSpace,
    size_t                                    workSpaceSizeInBytes,
    const void                                *beta,
    const FilterDescriptor                    dwDesc,
    void                                      *dw )
  {

    // TODO: implement this
  }


  auto findConvolutionBackwardDataAlgorithm (
    const Handle&                             handle,
    const FilterDescriptor&                   wDesc,
    const TensorDescriptor&                   dyDesc,
    const ConvolutionDescriptor&              convDesc,
    const TensorDescriptor&                   dxDesc,
    const int                                 requestedAlgoCount,
    int                                       *returnedAlgoCount,
    ConvolutionBwdDataAlgoPerf                *perfResults )
  {
    // TODO: implement this...
  }

  auto findConvolutionBackwardDataAlgorithmEx (
    const Handle&                             handle,
    const FilterDescriptor&                   wDesc,
    const void                                *w,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const ConvolutionDescriptor&              convDesc,
    const TensorDescriptor&                   dxDesc,
    void                                      *dx,
    const int                                 requestedAlgoCount,
    int                                       *returnedAlgoCount,
    ConvolutionBwdDataAlgoPerf                *perfResults,
    void                                      *workSpace,
    size_t                                    workSpaceSizeInBytes )
  {
    // TODO: implement this...
  }

  auto getConvolutionBackwardDataAlgorithm (
    const Handle&                             handle,
    const FilterDescriptor&                   wDesc,
    const TensorDescriptor&                   dyDesc,
    const ConvolutionDescriptor&              convDesc,
    const TensorDescriptor&                   dxDesc,
    const ConvolutionBwdDataPreference&       preference,
    size_t                                    memoryLimitInBytes,
    const ConvolutionBwdDataAlgo&             algo )
  {
    // TODO: implement this...
  }

  auto getConvolutionBackwardDataWorkspaceSize (
    const Handle&                             handle,
    const FilterDescriptor&                   wDesc,
    const TensorDescriptor&                   dyDesc,
    const ConvolutionDescriptor&              convDesc,
    const TensorDescriptor&                   dxDesc,
    const ConvolutionBwdDataAlgo&             algo,
    size_t                                    *sizeInBytes )
  {
    // TODO: implement this...
  }

  auto convolutionBackwardData (
    const Handle&                             handle,
    const void                                *alpha,
    const FilterDescriptor&                   wDesc,
    const void                                *w,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const ConvolutionDescriptor&              convDesc,
    const ConvolutionBwdDataAlgo&             algo,
    void                                      *workSpace,
    size_t                                    workSpaceSizeInBytes,
    const void                                *beta,
    const TensorDescriptor&                   dxDesc,
    void                                      *dx )
  {
    // TODO: implement this...
  }

  auto softmaxForward (
    const Handle&                             handle,
    const SoftmaxAlgorithm&                   algo,
    const SoftmaxMode&                        mode,
    const void                                *alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *beta,
    const TensorDescriptor&                   yDesc,
    void                                      *y )
  {
    // TODO: implement this...
  }

  auto softmaxBackward (
    const Handle&                             handle,
    const SoftmaxAlgorithm&                   algo,
    const SoftmaxMode&                        mode,
    const void                                *alpha,
    const TensorDescriptor&                   yDesc,
    const void                                *y,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const void                                *beta,
    const TensorDescriptor&                   dxDesc,
    void                                      *dx )
  {
    // TODO: implement this...
  }

  auto setPooling2dDescriptor (
    PoolingDescriptor                         poolingDesc,
    PoolingMode&                              mode,
    const NanPropagation&                     maxpoolingNanOpt,
    int                                       windowHeight,
    int                                       windowWidth,
    int                                       verticalPadding,
    int                                       horizontalPadding,
    int                                       verticalStride,
    int                                       horizontalStride )
  {
    // TODO: implement this...
  }

  auto getPooling2dDescriptor (
    const PoolingDescriptor                   poolingDesc,
    PoolingMode&                              mode,
    const NanPropagation&                     maxpoolingNanOpt,
    int                                       *windowHeight,
    int                                       *windowWidth,
    int                                       *verticalPadding,
    int                                       *horizontalPadding,
    int                                       *verticalStride,
    int                                       *horizontalStride )
  {
    // TODO: implement this...
  }

  auto setPoolingNdDescriptor (
    PoolingDescriptor                         poolingDesc,
    const PoolingMode&                        mode,
    const NanPropagation&                     maxpoolingNanOpt,
    int                                       nbDims,
    const int                                 windowDimA[],
    const int                                 paddingA[],
    const int                                 strideA[] )
  {
    // TODO: implement this...
  }

  auto getPoolingNdDescriptor (
    const PoolingDescriptor                   poolingDesc,
    int                                       nbDimsRequested,
    PoolingMode&                              mode,
    const NanPropagation&                     maxpoolingNanOpt,
    int                                       *nbDims,
    int                                       windowDimA[],
    int                                       paddingA[],
    int                                       strideA[] )
  {
    // TODO: implement this...
  }

  auto getPoolingNdForwardOutputDim (
    const PoolingDescriptor                   poolingDesc,
    const TensorDescriptor&                   inputTensorDesc,
    int                                       nbDims,
    int                                       outputTensorDimA[] )
  {
    // TODO: implement this...
  }

  auto getPooling2dForwardOutputDim (
    const PoolingDescriptor                   poolingDesc,
    const TensorDescriptor&                   inputTensorDesc,
    int                                       *n,
    int                                       *c,
    int                                       *h,
    int                                       *w )
  {
    // TODO: implement this...
  }

  auto poolingForward (
    const Handle&                             handle,
    const PoolingDescriptor                   poolingDesc,
    const void                                *alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *beta,
    const TensorDescriptor&                   yDesc,
    void                                      *y )
  {
    // TODO: implement this...
  }

  auto poolingBackward (
    const Handle&                             handle,
    const PoolingDescriptor                   poolingDesc,
    const void                                *alpha,
    const TensorDescriptor&                   yDesc,
    const void                                *y,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *beta,
    const TensorDescriptor&                   dxDesc,
    void                                      *dx )
  {
    // TODO: implement this...
  }

  auto activationForward (
    const Handle&                             handle,
    const ActivationDescriptor&               activationDesc,
    const float                               alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const float                               beta,
    const TensorDescriptor&                   yDesc,
    void                                      *y )
  {
    checkStatus(
      cudnnActivationForward(
        handle,
        activationDesc,
        &alpha,
        xDesc,
        x,
        &beta,
        yDesc,
        y));
  }

  auto activationBackward (
    const Handle&                             handle,
    const ActivationDescriptor&               activationDesc,
    const float                               alpha,
    const TensorDescriptor&                   yDesc,
    const long long                           y,
    const TensorDescriptor&                   dyDesc,
    const long long                           dy,
    const TensorDescriptor&                   xDesc,
    const long long                           x,
    const float                               beta,
    const TensorDescriptor&                   dxDesc,
    const long long                           dx )
  {
    checkStatus(
      cudnnActivationBackward(
        handle,
        activationDesc,
        &alpha,
        yDesc,
        (void*) y,
        dyDesc,
        (void*) dy,
        xDesc,
        (void*) x,
        &beta,
        dxDesc,
        (void*) dx));
  }

  auto setActivationDescriptor (
    const ActivationDescriptor&               activationDesc,
    const ActivationMode&                     mode,
    const NanPropagation&                     reluNanOpt,
    double                                    reluCeiling )
  {
    checkStatus(
      cudnnSetActivationDescriptor(
        activationDesc,
        mode,
        reluNanOpt,
        reluCeiling));
  }

  auto getActivationDescriptor (
    const ActivationDescriptor&               activationDesc,
    const ActivationMode&                     mode,
    const NanPropagation&                     reluNanOpt,
    double*                                   reluCeiling )
  {
    // TODO: implement this...
  }

  auto setLRNDescriptor (
    const LRNDescriptor&                      normDesc,
    unsigned                                  lrnN,
    double                                    lrnAlpha,
    double                                    lrnBeta,
    double                                    lrnK )
  {
    // TODO: implement this...
  }

  auto getLRNDescriptor (
    const LRNDescriptor&                      normDesc,
    unsigned*                                 lrnN,
    double*                                   lrnAlpha,
    double*                                   lrnBeta,
    double*                                   lrnK )
  {
    // TODO: implement this...
  }

  auto lrnCrossChannelForward (
    const Handle&                             handle,
    const LRNDescriptor&                      normDesc,
    const LRNMode&                            lrnMode,
    const void*                               alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *beta,
    const TensorDescriptor&                   yDesc,
    void                                      *y )
  {
    // TODO: implement this...
  }

  auto lrnCrossChannelBackward (
    const Handle&                             handle,
    const LRNDescriptor&                      normDesc,
    const LRNMode&                            lrnMode,
    const void*                               alpha,
    const TensorDescriptor&                   yDesc,
    const void                                *y,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *beta,
    const TensorDescriptor&                   dxDesc,
    void                                      *dx)
  {
    // TODO: implement this...
  }

  auto divisiveNormalizationForward (
    const Handle&                             handle,
    const LRNDescriptor&                      normDesc,
    const DivNormMode&                        mode,
    const void                                *alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *means,
    void                                      *temp,
    void                                      *temp2,
    const void                                *beta,
    const TensorDescriptor&                   yDesc,
    void                                      *y )
  {
    // TODO: implement this...
  }

  auto divisiveNormalizationBackward (
    const Handle&                             handle,
    const LRNDescriptor&                      normDesc,
    const DivNormMode&                        mode,
    const void                                *alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *means,
    const void                                *dy,
    void                                      *temp,
    void                                      *temp2,
    const void                                *beta,
    const TensorDescriptor&                   dXdMeansDesc,
    void                                      *dx,
    void                                      *dMeans )
    {
      // TODO: implement this...
    } /* output means differential, can be NULL*/

  auto batchNormalizationForwardInference (
    const Handle&                             handle,
    const BatchNormMode&                      mode,
    const void                                *alpha,
    const void                                *beta,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const TensorDescriptor&                   yDesc,
    void                                      *y,
    const TensorDescriptor&                   bnScaleBiasMeanVarDesc,
    const void                                *bnScale,
    const void                                *bnBias,
    const void                                *estimatedMean,
    const void                                *estimatedVariance,
    double                                    epsilon )
  {
    // TODO: implement this...
  }

  auto batchNormalizationForwardTraining (
    const Handle&                             handle,
    const BatchNormMode&                      mode,
    const void                                *alpha,
    const void                                *beta,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const TensorDescriptor&                   yDesc,
    void                                      *y,
    const TensorDescriptor&                   bnScaleBiasMeanVarDesc,
    const void                                *bnScale,
    const void                                *bnBias,
    double                                    exponentialAverageFactor,
    void                                      *resultRunningMean,
    void                                      *resultRunningVariance,
    double                                    epsilon,
    void                                      *resultSaveMean,
    void                                      *resultSaveInvVariance )
  {
    // TODO: implement this...
  }

  auto batchNormalizationBackward (
    const Handle&                             handle,
    const BatchNormMode&                      mode,
    const void                                *alphaDataDiff,
    const void                                *betaDataDiff,
    const void                                *alphaParamDiff,
    const void                                *betaParamDiff,
    const TensorDescriptor&                   xDesc, /* same desc for x, dx, dy*/
    const void                                *x,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const TensorDescriptor&                   dxDesc,
    void                                      *dx,
    /* Shared tensor desc for the 4 tensors below */
    const TensorDescriptor&                   dBnScaleBiasDesc,
    const void                                *bnScale, /* bnBias doesn't affect backpropagation*/
    /* scale and bias diff are not backpropagated below this layer */
    void                                      *dBnScaleResult,
    void                                      *dBnBiasResult,
    /* Same epsilon as forward pass */
    double                                    epsilon,

    /* Optionally cached intermediate results from
       forward pass */
    const void                                *savedMean,
    const void                                *savedInvVariance )
  {
    // TODO: implement this...
  }

  auto deriveBNTensorDescriptor (
    TensorDescriptor&                         derivedBnDesc,
    const TensorDescriptor&                   xDesc,
    const BatchNormMode&                      mode )
  {
    // TODO: implement this...
  }

  auto setRNNDescriptor (
    const RNNDescriptor&                      rnnDesc,
    int                                       hiddenSize,
    int                                       numLayers,
    DropoutDescriptor&                        dropoutDesc,
    const RNNInputMode&                       inputMode,
    const DirectionMode&                      direction,
    const RNNMode&                            mode,
    DataType&                                 dataType)
  {
    // TODO: implement this...
  }

  auto getRNNWorkspaceSize (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const int                                 seqLength,
    const TensorDescriptor&                   xDesc,
    size_t                                    *sizeInBytes)
  {
    // TODO: implement this...
  }

  auto getRNNTrainingReserveSize (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const int                                 seqLength,
    const TensorDescriptor&                   xDesc,
    size_t                                    *sizeInBytes)
  {
    // TODO: implement this...
  }


  auto getRNNParamsSize (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const TensorDescriptor&                   xDesc,
    size_t                                    *sizeInBytes,
    DataType&                                 dataType)
  {
    // TODO: implement this...
  }

  auto getRNNLinLayerMatrixParams (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const int                                 layer,
    const TensorDescriptor&                   xDesc,
    const FilterDescriptor&                   wDesc,
    const void                                *w,
    const int                                 linLayerID,
    FilterDescriptor&                         linLayerMatDesc,
    void                                      **linLayerMat)
  {
    // TODO: implement this...
  }

  auto getRNNLinLayerBiasParams (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const int                                 layer,
    const TensorDescriptor&                   xDesc,
    const FilterDescriptor&                   wDesc,
    const void                                *w,
    const int                                 linLayerID,
    FilterDescriptor&                         linLayerBiasDesc,
    void                                      **linLayerBias)
  {
    // TODO: implement this...
  }

  auto rnnForwardInference (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const int                                 seqLength,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const TensorDescriptor&                   hxDesc,
    const void                                *hx,
    const TensorDescriptor&                   cxDesc,
    const void                                *cx,
    const FilterDescriptor&                   wDesc,
    const void                                *w,
    const TensorDescriptor&                   yDesc,
    void                                      *y,
    const TensorDescriptor&                   hyDesc,
    void                                      *hy,
    const TensorDescriptor&                   cyDesc,
    void                                      *cy,
    void                                      *workspace,
    size_t                                    workSpaceSizeInBytes)
  {
    // TODO: implement this...
  }

  auto rnnForwardTraining (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const int                                 seqLength,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const TensorDescriptor&                   hxDesc,
    const void                                *hx,
    const TensorDescriptor&                   cxDesc,
    const void                                *cx,
    const FilterDescriptor&                   wDesc,
    const void                                *w,
    const TensorDescriptor&                   yDesc,
    void                                      *y,
    const TensorDescriptor&                   hyDesc,
    void                                      *hy,
    const TensorDescriptor&                   cyDesc,
    void                                      *cy,
    void                                      *workspace,
    size_t                                    workSpaceSizeInBytes,
    void                                      *reserveSpace,
    size_t                                    reserveSpaceSizeInBytes)
  {
    // TODO: implement this...
  }

  auto rnnBackwardData (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const int                                 seqLength,
    const TensorDescriptor&                   yDesc,
    const void                                *y,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const TensorDescriptor&                   dhyDesc,
    const void                                *dhy,
    const TensorDescriptor&                   dcyDesc,
    const void                                *dcy,
    const FilterDescriptor&                   wDesc,
    const void                                *w,
    const TensorDescriptor&                   hxDesc,
    const void                                *hx,
    const TensorDescriptor&                   cxDesc,
    const void                                *cx,
    const TensorDescriptor&                   dxDesc,
    void                                      *dx,
    const TensorDescriptor&                   dhxDesc,
    void                                      *dhx,
    const TensorDescriptor&                   dcxDesc,
    void                                      *dcx,
    void                                      *workspace,
    size_t                                    workSpaceSizeInBytes,
    const void                                *reserveSpace,
    size_t                                    reserveSpaceSizeInBytes )
  {
    // TODO: implement this...
  }

  auto rnnBackwardWeights (
    const Handle&                             handle,
    const RNNDescriptor&                      rnnDesc,
    const int                                 seqLength,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const TensorDescriptor&                   hxDesc,
    const void                                *hx,
    const TensorDescriptor&                   yDesc,
    const void                                *y,
    const void                                *workspace,
    size_t                                    workSpaceSizeInBytes,
    const FilterDescriptor&                   dwDesc,
    void                                      *dw,
    const void                                *reserveSpace,
    size_t                                    reserveSpaceSizeInBytes )
  {
    // TODO: implement this...
  }

  auto dropoutGetStatesSize (
    const Handle&                             handle,
    size_t                                    *sizeInBytes)
  {
    // TODO: implement this...
  }

  auto dropoutGetReserveSpaceSize (
    TensorDescriptor&                         xDesc,
    size_t                                    *sizeInBytes )
  {
    // TODO: implement this...
  }

  auto setDropoutDescriptor (
    const DropoutDescriptor&                  dropoutDesc,
    const Handle&                             handle,
    float                                     dropout,
    void                                      *states,
    size_t                                    stateSizeInBytes,
    unsigned long long                        seed )
  {
    // TODO: implement this...
  }

  auto dropoutForward (
    const Handle&                             handle,
    const DropoutDescriptor&                  dropoutDesc,
    const TensorDescriptor&                   xdesc,
    const void                                *x,
    const TensorDescriptor&                   ydesc,
    void                                      *y,
    void                                      *reserveSpace,
    size_t                                    reserveSpaceSizeInBytes )
  {
    // TODO: implement this...
  }

  auto dropoutBackward (
    const Handle&                             handle,
    const DropoutDescriptor&                  dropoutDesc,
    const TensorDescriptor&                   dydesc,
    const void                                *dy,
    const TensorDescriptor&                   dxdesc,
    void                                      *dx,
    void                                      *reserveSpace,
    size_t                                    reserveSpaceSizeInBytes )
  {
    // TODO: implement this...
  }

  auto setSpatialTransformerNdDescriptor (
    const SpatialTransformerDescriptor&       stDesc,
    const SamplerType&                        samplerType,
    DataType&                                 dataType,
    const int                                 nbDims,
    const int                                 dimA[])
  {
    // TODO: implement this...
  }

  auto spatialTfGridGeneratorForward (
    const Handle&                             handle,
    const SpatialTransformerDescriptor&       stDesc,
    const void                                *theta,
    void                                      *grid)
  {
    // TODO: implement this...
  }

  auto spatialTfGridGeneratorBackward (
    const Handle&                             handle,
    const SpatialTransformerDescriptor&       stDesc,
    const void                                *dgrid,
    void                                      *dtheta)
  {
    // TODO: implement this...
  }

  auto spatialTfSamplerForward (
    const Handle&                             handle,
    const SpatialTransformerDescriptor&       stDesc,
    const void                                *alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *grid,
    const void                                *beta,
    TensorDescriptor&                         yDesc,
    void                                      *y)
  {
    // TODO: implement this...
  }

  auto spatialTfSamplerBackward (
    const Handle&                             handle,
    const SpatialTransformerDescriptor&       stDesc,
    const void                                *alpha,
    const TensorDescriptor&                   xDesc,
    const void                                *x,
    const void                                *beta,
    const TensorDescriptor&                   dxDesc,
    void                                      *dx,
    const void                                *alphaDgrid,
    const TensorDescriptor&                   dyDesc,
    const void                                *dy,
    const void                                *grid,
    const void                                *betaDgrid,
    void                                      *dgrid)
  {
    // TODO: implement this...
  }

}

PYBIND11_PLUGIN(pycudnn) {
  using namespace PyCuDNN;

  py::module m("pycudnn", R"docstring(

    .. currentmodule:: pycudnn

    Classes:
    --------

    .. autosummary::
       :toctree: .generate

       ActivationDescriptor
       ConvolutionBwdDataAlgoPerf
       ConvolutionDescriptor
       FilterDescriptor
       DropoutDescriptor
       Handle
       OpTensorDescriptor
       PoolingDescriptor
       LRNDescriptor
       RNNDescriptor
       SpatialTransformerDescriptor
       TensorDescriptor
       ConvolutionBwdFilterAlgoPerf
       ConvolutionFwdAlgoPerf


    Enums:
    ------

    .. autosummary::
       :toctree: .generate

       ActivationMode
       BatchNormMode
       ConvolutionBwdDataAlgo
       ConvolutionBwdDataPreference
       ConvolutionBwdFilterAlgo
       ConvolutionBwdFilterPreference
       ConvolutionFwdAlgo
       ConvolutionFwdPreference
       ConvolutionMode
       DataType
       DirectionMode
       DivNormMode
       LRNMode
       NanPropagation
       OpTensorOp
       PoolingMode
       RNNInputMode
       RNNMode
       SamplerType
       SoftmaxAlgorithm
       SoftmaxMode
       Status
       TensorFormat


    Functions:
    ----------

    .. autosummary::
       :toctree: .generate

       get_version
       get_error_string
       set_tensor_4d_descriptor
       set_tensor_4d_descriptor_ex
       get_tensor_4d_descriptor
       set_tensor_nd_descriptor
       get_tensor_nd_descriptor
       transform_tensor
       add_tensor
       op_tensor
       set_tensor
       scale_tensor
       set_filter_4d_descriptor
       get_filter_4d_descriptor
       set_filter_nd_descriptor
       get_filter_nd_descriptor
       set_convolution_2d_descriptor
       get_convolution_2d_descriptor
       get_convolution_2d_forward_output_dim
       set_convolution_nd_descriptor
       get_convolution_nd_descriptor
       get_convolution_nd_forward_output_dim
       find_convolution_forward_algorithm
       find_convolution_forward_algorithm_ex
       get_convolution_forward_algorithm
       get_convolution_forward_workspace_size
       convolution_forward
       convolution_backward_bias
       find_convolution_backward_filter_algorithm
       find_convolution_backward_filter_algorithm_ex
       activation_forward
       activation_backward
       set_activation_descriptor



    )docstring");

  py::register_exception<PyCuDNN::Exception>(m, "Exception");

  py::class_<ActivationDescriptor>(m, "ActivationDescriptor")
    .def(py::init<>());

  py::class_<ConvolutionBwdDataAlgoPerf>(m, "ConvolutionBwdDataAlgoPerf")
    .def_property_readonly("algo",
      [](const ConvolutionBwdDataAlgoPerf& self) { return self.algo; })
    .def_property_readonly("status",
      [](const ConvolutionBwdDataAlgoPerf& self) { return self.status; })
    .def_property_readonly("time",
      [](const ConvolutionBwdDataAlgoPerf& self) { return self.time; })
    .def_property_readonly("memory",
      [](const ConvolutionBwdDataAlgoPerf& self) { return self.memory; });

  py::class_<ConvolutionDescriptor>(m, "ConvolutionDescriptor")
    .def(py::init<>());

  py::class_<FilterDescriptor>(m, "FilterDescriptor")
    .def(py::init<>());

  py::class_<DropoutDescriptor>(m, "DropoutDescriptor")
    .def(py::init<>());

  py::class_<Handle>(m, "Handle")
    .def(py::init<>());

  py::class_<OpTensorDescriptor>(m, "OpTensorDescriptor")
    .def(py::init<>());

  py::class_<PoolingDescriptor>(m, "PoolingDescriptor")
    .def(py::init<>());

  py::class_<LRNDescriptor>(m, "LRNDescriptor")
    .def(py::init<>());

  py::class_<RNNDescriptor>(m, "RNNDescriptor")
    .def(py::init<>());

  py::class_<SpatialTransformerDescriptor>(m, "SpatialTransformerDescriptor")
    .def(py::init<>());

  py::class_<TensorDescriptor>(m, "TensorDescriptor")
    .def(py::init<>());

  py::enum_<ActivationMode>(m, "ActivationMode")
    .value("CUDNN_ACTIVATION_SIGMOID",
        CUDNN_ACTIVATION_SIGMOID)
    .value("CUDNN_ACTIVATION_RELU",
        CUDNN_ACTIVATION_RELU)
    .value("CUDNN_ACTIVATION_TANH",
        CUDNN_ACTIVATION_TANH)
    .value("CUDNN_ACTIVATION_CLIPPED_RELU",
        CUDNN_ACTIVATION_CLIPPED_RELU)
    .export_values();

  py::enum_<BatchNormMode>(m, "BatchNormMode")
    .value("CUDNN_BATCHNORM_PER_ACTIVATION",
        CUDNN_BATCHNORM_PER_ACTIVATION)
    .value("CUDNN_BATCHNORM_SPATIAL",
        CUDNN_BATCHNORM_SPATIAL)
    .export_values();

  py::enum_<ConvolutionBwdDataAlgo>(m, "ConvolutionBwdDataAlgo")
    .value("CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
    .value("CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
    .value("CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
    .value("CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
    .value("CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
    .value("CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
    .export_values();

  py::enum_<ConvolutionBwdDataPreference>(m, "ConvolutionBwdDataPreference")
    .value("CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",
        CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE)
    .value("CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST)
    .value("CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT)
    .export_values();

  py::enum_<ConvolutionBwdFilterAlgo>(m, "ConvolutionBwdFilterAlgo")
    .value("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)
    .value("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
    .value("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
    .value("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
    .value("CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
    .export_values();

  py::class_<ConvolutionBwdFilterAlgoPerf>(m, "ConvolutionBwdFilterAlgoPerf")
    .def_property_readonly("algo",
      [](const ConvolutionBwdFilterAlgoPerf& self) { return self.algo; })
    .def_property_readonly("status",
      [](const ConvolutionBwdFilterAlgoPerf& self) { return self.status; })
    .def_property_readonly("time",
      [](const ConvolutionBwdFilterAlgoPerf& self) { return self.time; })
    .def_property_readonly("memory",
      [](const ConvolutionBwdFilterAlgoPerf& self) { return self.memory; });

  py::enum_<ConvolutionBwdFilterPreference>(m, "ConvolutionBwdFilterPreference")
    .value("CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",
        CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE)
    .value("CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)
    .value("CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT",
        CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
    .export_values();

  py::enum_<ConvolutionFwdAlgo>(m, "ConvolutionFwdAlgo")
    .value("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
    .value("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
    .value("CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
    .value("CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
    .value("CUDNN_CONVOLUTION_FWD_ALGO_FFT",
        CUDNN_CONVOLUTION_FWD_ALGO_FFT)
    .value("CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
    .value("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
    .value("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
    .export_values();

  py::class_<ConvolutionFwdAlgoPerf>(m, "ConvolutionFwdAlgoPerf")
    .def_property_readonly("algo",
      [](const ConvolutionFwdAlgoPerf& self) { return self.algo; })
    .def_property_readonly("status",
      [](const ConvolutionFwdAlgoPerf& self) { return self.status; })
    .def_property_readonly("time",
      [](const ConvolutionFwdAlgoPerf& self) { return self.time; })
    .def_property_readonly("memory",
      [](const ConvolutionFwdAlgoPerf& self) { return self.memory; });

  py::enum_<ConvolutionFwdPreference>(m, "ConvolutionFwdPreference")
    .value("CUDNN_CONVOLUTION_FWD_NO_WORKSPACE",
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE)
    .value("CUDNN_CONVOLUTION_FWD_PREFER_FASTEST",
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST)
    .value("CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
    .export_values();

  py::enum_<ConvolutionMode>(m, "ConvolutionMode")
    .value("CUDNN_CONVOLUTION",
        CUDNN_CONVOLUTION)
    .value("CUDNN_CROSS_CORRELATION",
        CUDNN_CROSS_CORRELATION)
    .export_values();

  py::enum_<DataType>(m, "DataType")
    .value("CUDNN_DATA_FLOAT",
        CUDNN_DATA_FLOAT)
    .value("CUDNN_DATA_DOUBLE",
        CUDNN_DATA_DOUBLE)
    .export_values();

  py::enum_<DirectionMode>(m, "DirectionMode")
    .value("CUDNN_UNIDIRECTIONAL",
        CUDNN_UNIDIRECTIONAL)
    .value("CUDNN_BIDIRECTIONAL",
        CUDNN_BIDIRECTIONAL)
    .export_values();

  py::enum_<DivNormMode>(m, "DivNormMode")
    .value("CUDNN_DIVNORM_PRECOMPUTED_MEANS",
        CUDNN_DIVNORM_PRECOMPUTED_MEANS)
    .export_values();

  py::enum_<LRNMode>(m, "LRNMode")
    .value("CUDNN_LRN_CROSS_CHANNEL_DIM1",
        CUDNN_LRN_CROSS_CHANNEL_DIM1)
    .export_values();

  py::enum_<NanPropagation>(m, "NanPropagation")
    .value("CUDNN_NOT_PROPAGATE_NAN",
        CUDNN_NOT_PROPAGATE_NAN)
    .value("CUDNN_PROPAGATE_NAN",
        CUDNN_PROPAGATE_NAN)
    .export_values();

  py::enum_<OpTensorOp>(m, "OpTensorOp")
    .value("CUDNN_OP_TENSOR_ADD",
        CUDNN_OP_TENSOR_ADD)
    .value("CUDNN_OP_TENSOR_MUL",
        CUDNN_OP_TENSOR_MUL)
    .value("CUDNN_OP_TENSOR_MIN",
        CUDNN_OP_TENSOR_MIN)
    .value("CUDNN_OP_TENSOR_MAX",
        CUDNN_OP_TENSOR_MAX)
    .export_values();

  py::enum_<PoolingMode>(m, "PoolingMode")
    .value("CUDNN_POOLING_MAX",
        CUDNN_POOLING_MAX)
    .value("CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
    .value("CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
    .export_values();

  py::enum_<RNNInputMode>(m, "RNNInputMode")
    .value("CUDNN_LINEAR_INPUT",
        CUDNN_LINEAR_INPUT)
    .value("CUDNN_SKIP_INPUT",
        CUDNN_SKIP_INPUT)
    .export_values();

  py::enum_<RNNMode>(m, "RNNMode")
    .value("CUDNN_RNN_RELU",
        CUDNN_RNN_RELU)
    .value("CUDNN_RNN_TANH",
        CUDNN_RNN_TANH)
    .value("CUDNN_LSTM",
        CUDNN_LSTM)
    .value("CUDNN_GRU",
        CUDNN_GRU)
    .export_values();

  py::enum_<SamplerType>(m, "SamplerType")
    .value("CUDNN_SAMPLER_BILINEAR",
        CUDNN_SAMPLER_BILINEAR)
    .export_values();

  py::enum_<SoftmaxAlgorithm>(m, "SoftmaxAlgorithm")
    .value("CUDNN_SOFTMAX_FAST",
        CUDNN_SOFTMAX_FAST)
    .value("CUDNN_SOFTMAX_ACCURATE",
        CUDNN_SOFTMAX_ACCURATE)
    .value("CUDNN_SOFTMAX_LOG",
        CUDNN_SOFTMAX_LOG)
    .export_values();

  py::enum_<SoftmaxMode>(m, "SoftmaxMode")
    .value("CUDNN_SOFTMAX_MODE_INSTANCE",
        CUDNN_SOFTMAX_MODE_INSTANCE)
    .value("CUDNN_SOFTMAX_MODE_CHANNEL",
        CUDNN_SOFTMAX_MODE_CHANNEL)
    .export_values();

  py::enum_<Status>(m, "Status")
    .value("CUDNN_STATUS_SUCCESS",
        CUDNN_STATUS_SUCCESS)
    .value("CUDNN_STATUS_NOT_INITIALIZED",
        CUDNN_STATUS_NOT_INITIALIZED)
    .value("CUDNN_STATUS_ALLOC_FAILED",
        CUDNN_STATUS_ALLOC_FAILED)
    .value("CUDNN_STATUS_BAD_PARAM",
        CUDNN_STATUS_BAD_PARAM)
    .value("CUDNN_STATUS_ARCH_MISMATCH",
        CUDNN_STATUS_ARCH_MISMATCH)
    .value("CUDNN_STATUS_MAPPING_ERROR",
        CUDNN_STATUS_MAPPING_ERROR)
    .value("CUDNN_STATUS_EXECUTION_FAILED",
        CUDNN_STATUS_EXECUTION_FAILED)
    .value("CUDNN_STATUS_INTERNAL_ERROR",
        CUDNN_STATUS_INTERNAL_ERROR)
    .value("CUDNN_STATUS_NOT_SUPPORTED",
        CUDNN_STATUS_NOT_SUPPORTED)
    .value("CUDNN_STATUS_LICENSE_ERROR",
        CUDNN_STATUS_LICENSE_ERROR)
    .export_values();

  py::enum_<TensorFormat>(m, "TensorFormat")
    .value("CUDNN_TENSOR_NCHW",
        CUDNN_TENSOR_NCHW)
    .value("CUDNN_TENSOR_NHWC",
        CUDNN_TENSOR_NHWC)
    .export_values();

  m.def("get_version", &getVersion);
  m.def("get_error_string", &getErrorString);
  // Note: there's currently an compile issue with cudaStream_t type
  // m.def("set_stream", [](const Handle& handle, cudaStream_t streamId) {
  //   checkStatus(cudnnSetStream(handle, streamId));
  // });
  // m.def("get_stream", [](const Handle& handle) {
  //   cudaStream_t streamId;
  //   checkStatus(cudnnGetStream(handle, &streamId));
  //   return streamId;
  // });
  // TODO: fix CUDA streams issue
  m.def("set_tensor_4d_descriptor", &setTensor4dDescriptor);
  m.def("set_tensor_4d_descriptor_ex", &setTensor4dDescriptorEx);
  m.def("get_tensor_4d_descriptor", &getTensor4dDescriptor);
  m.def("set_tensor_nd_descriptor", &setTensorNdDescriptor);
  m.def("get_tensor_nd_descriptor", &getTensorNdDescriptor);
  m.def("transform_tensor", &transformTensor);
  m.def("add_tensor", &addTensor);
  m.def("op_tensor", &opTensor);
  m.def("set_tensor", &setTensor);
  m.def("scale_tensor", &scaleTensor);
  m.def("set_filter_4d_descriptor", &setFilter4dDescriptor);
  m.def("get_filter_4d_descriptor", &getFilter4dDescriptor);
  m.def("set_filter_nd_descriptor", &setFilterNdDescriptor);
  m.def("get_filter_nd_descriptor", &getFilterNdDescriptor);
  m.def("set_convolution_2d_descriptor", &setConvolution2dDescriptor);
  m.def("get_convolution_2d_descriptor", &getConvolution2dDescriptor);
  m.def("get_convolution_2d_forward_output_dim", &getConvolution2dForwardOutputDim);
  m.def("set_convolution_nd_descriptor", &setConvolutionNdDescriptor);
  m.def("get_convolution_nd_descriptor", &getConvolutionNdDescriptor);
  m.def("get_convolution_nd_forward_output_dim", &getConvolutionNdForwardOutputDim);
  m.def("find_convolution_forward_algorithm", &findConvolutionForwardAlgorithm);
  m.def("find_convolution_forward_algorithm_ex", &findConvolutionForwardAlgorithmEx);
  m.def("get_convolution_forward_algorithm", &getConvolutionForwardAlgorithm);
  m.def("get_convolution_forward_workspace_size", &getConvolutionForwardAlgorithm);
  m.def("convolution_forward", &convolutionForward);
  m.def("convolution_backward_bias", &convolutionBackwardBias);
  m.def("find_convolution_backward_filter_algorithm", &findConvolutionBackwardFilterAlgorithm);
  m.def("find_convolution_backward_filter_algorithm_ex", &findConvolutionBackwardFilterAlgorithmEx);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_convolution_backward_filter_workspace_size", &getConvolutionBackwardFilterWorkspaceSize);
  // TODO: implement corresponding function and uncomment:
  //       m.def("convolution_backward_filter", &convolutionBackwardFilter);
  // TODO: implement corresponding function and uncomment:
  //       m.def("find_convolution_backward_data_algorithm", &findConvolutionBackwardDataAlgorithm);
  // TODO: implement corresponding function and uncomment:
  //       m.def("find_convolution_backward_data_algorithm_ex", &findConvolutionBackwardDataAlgorithmEx);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_convolution_backward_data_algorithm", &getConvolutionBackwardDataAlgorithm);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_convolution_backward_data_workspace_size", &getConvolutionBackwardDataWorkspaceSize);
  // TODO: implement corresponding function and uncomment:
  //       m.def("convolution_backward_data", &convolutionBackwardData);
  // TODO: implement corresponding function and uncomment:
  //       m.def("softmax_forward", &softmaxForward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("softmax_backward", &softmaxBackward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("set_pooling2d_descriptor", &setPooling2dDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_pooling2d_descriptor", &getPooling2dDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("set_pooling_nd_descriptor", &setPoolingNdDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_pooling_nd_descriptor", &getPoolingNdDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_pooling_nd_forward_output_dim", &getPoolingNdForwardOutputDim);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_pooling2d_forward_output_dim", &getPooling2dForwardOutputDim);
  // TODO: implement corresponding function and uncomment:
  //       m.def("pooling_forward", &poolingForward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("pooling_backward", &poolingBackward);
  m.def("activation_forward", &activationForward);
  m.def("activation_backward", &activationBackward);
  m.def("set_activation_descriptor", &setActivationDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_activation_descriptor", &getActivationDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("set_lrn_descriptor", &setLRNDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_lrn_descriptor", &getLRNDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("lrn_cross_channel_forward", &lrnCrossChannelForward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("lrn_cross_channel_backward", &lrnCrossChannelBackward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("divisive_normalization_forward", &divisiveNormalizationForward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("divisive_normalization_backward", &divisiveNormalizationBackward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("batch_normalization_forward_inference", &batchNormalizationForwardInference);
  // TODO: implement corresponding function and uncomment:
  //       m.def("batch_normalization_forward_training", &batchNormalizationForwardTraining);
  // TODO: implement corresponding function and uncomment:
  //       m.def("batch_normalization_backward", &batchNormalizationBackward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("derive_bn_tensor_descriptor", &deriveBNTensorDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("set_rnn_descriptor", &setRNNDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_rnn_workspace_size", &getRNNWorkspaceSize);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_rnn_training_reserve_size", &getRNNTrainingReserveSize);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_rnn_params_size", &getRNNParamsSize);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_rnn_lin_layer_matrix_params", &getRNNLinLayerMatrixParams);
  // TODO: implement corresponding function and uncomment:
  //       m.def("get_rnn_lin_layer_bias_params", &getRNNLinLayerBiasParams);
  // TODO: implement corresponding function and uncomment:
  //       m.def("rnn_forward_inference", &rnnForwardInference);
  // TODO: implement corresponding function and uncomment:
  //       m.def("rnn_forward_training", &rnnForwardTraining);
  // TODO: implement corresponding function and uncomment:
  //       m.def("rnn_backward_data", &rnnBackwardData);
  // TODO: implement corresponding function and uncomment:
  //       m.def("rnn_backward_weights", &rnnBackwardWeights);
  // TODO: implement corresponding function and uncomment:
  //       m.def("dropout_get_states_size", &dropoutGetStatesSize);
  // TODO: implement corresponding function and uncomment:
  //       m.def("dropout_get_reserve_space_size", &dropoutGetReserveSpaceSize);
  // TODO: implement corresponding function and uncomment:
  //       m.def("set_dropout_descriptor", &setDropoutDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("dropout_forward", &dropoutForward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("dropout_backward", &dropoutBackward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("set_spatial_transformer_nd_descriptor", &setSpatialTransformerNdDescriptor);
  // TODO: implement corresponding function and uncomment:
  //       m.def("spatial_tf_grid_generator_forward", &spatialTfGridGeneratorForward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("spatial_tf_grid_generator_backward", &spatialTfGridGeneratorBackward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("spatial_tf_sampler_forward", &spatialTfSamplerForward);
  // TODO: implement corresponding function and uncomment:
  //       m.def("spatial_tf_sampler_backward", &spatialTfSamplerBackward);

  return m.ptr();
}
