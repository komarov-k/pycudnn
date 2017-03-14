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
#include "PyCuDNNHandle.hpp"
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

PYBIND11_PLUGIN(pycudnn) {

	py::module m("pycudnn", "Python interface to NVIDIA CuDNN library");

	using namespace PyCuDNN;

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

	py::class_<DropoutDescriptor>(m, "DropoutDescriptor")
		.def(py::init<>());

	py::class_<Handle>(m, "Handle")
		.def(py::init<>());

	py::class_<OpTensorDescriptor>(m, "OpTensorDescriptor")
		.def(py::init<>());

	py::class_<PoolingDescriptor>(m, "PoolingDescriptor")
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
				CUDNN_ACTIVATION_CLIPPED_RELU);

	py::enum_<BatchNormMode>(m, "BatchNormMode")
		.value("CUDNN_BATCHNORM_PER_ACTIVATION",
				CUDNN_BATCHNORM_PER_ACTIVATION)
		.value("CUDNN_BATCHNORM_SPATIAL",
				CUDNN_BATCHNORM_SPATIAL);

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
				CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);

	py::enum_<ConvolutionBwdDataPreference>(m, "ConvolutionBwdDataPreference")
		.value("CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",
				CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE)
		.value("CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST)
		.value("CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",
				CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT);

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
				CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);

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
				CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT);

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
				CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);

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
				CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT);

	py::enum_<ConvolutionMode>(m, "ConvolutionMode")
		.value("CUDNN_CONVOLUTION",
				CUDNN_CONVOLUTION)
		.value("CUDNN_CROSS_CORRELATION",
				CUDNN_CROSS_CORRELATION);

	py::enum_<DataType>(m, "DataType")
		.value("CUDNN_DATA_FLOAT",
				CUDNN_DATA_FLOAT)
		.value("CUDNN_DATA_DOUBLE",
				CUDNN_DATA_DOUBLE);

	py::enum_<DirectionMode>(m, "DirectionMode")
		.value("CUDNN_UNIDIRECTIONAL",
				CUDNN_UNIDIRECTIONAL)
		.value("CUDNN_BIDIRECTIONAL",
				CUDNN_BIDIRECTIONAL);

	py::enum_<DivNormMode>(m, "DivNormMode")
		.value("CUDNN_DIVNORM_PRECOMPUTED_MEANS",
				CUDNN_DIVNORM_PRECOMPUTED_MEANS);

	py::enum_<LRNMode>(m, "LRNMode")
		.value("CUDNN_LRN_CROSS_CHANNEL_DIM1",
				CUDNN_LRN_CROSS_CHANNEL_DIM1);

	py::enum_<NanPropagation>(m, "NanPropagation")
		.value("CUDNN_NOT_PROPAGATE_NAN",
				CUDNN_NOT_PROPAGATE_NAN)
		.value("CUDNN_PROPAGATE_NAN",
				CUDNN_PROPAGATE_NAN);

	py::enum_<OpTensorOp>(m, "OpTensorOp")
		.value("CUDNN_OP_TENSOR_ADD",
				CUDNN_OP_TENSOR_ADD)
		.value("CUDNN_OP_TENSOR_MUL",
				CUDNN_OP_TENSOR_MUL)
		.value("CUDNN_OP_TENSOR_MIN",
				CUDNN_OP_TENSOR_MIN)
		.value("CUDNN_OP_TENSOR_MAX",
				CUDNN_OP_TENSOR_MAX);

	py::enum_<PoolingMode>(m, "PoolingMode")
		.value("CUDNN_POOLING_MAX",
				CUDNN_POOLING_MAX)
		.value("CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",
				CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
		.value("CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",
				CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);

	py::enum_<RNNInputMode>(m, "RNNInputMode")
		.value("CUDNN_LINEAR_INPUT",
				CUDNN_LINEAR_INPUT)
		.value("CUDNN_SKIP_INPUT",
				CUDNN_SKIP_INPUT);

	py::enum_<RNNMode>(m, "RNNMode")
		.value("CUDNN_RNN_RELU",
				CUDNN_RNN_RELU)
		.value("CUDNN_RNN_TANH",
				CUDNN_RNN_TANH)
		.value("CUDNN_LSTM",
				CUDNN_LSTM)
		.value("CUDNN_GRU",
				CUDNN_GRU);

	py::enum_<SamplerType>(m, "SamplerType")
		.value("CUDNN_SAMPLER_BILINEAR",
				CUDNN_SAMPLER_BILINEAR);

	py::enum_<SoftmaxAlgorithm>(m, "SoftmaxAlgorithm")
		.value("CUDNN_SOFTMAX_FAST",
				CUDNN_SOFTMAX_FAST)
		.value("CUDNN_SOFTMAX_ACCURATE",
				CUDNN_SOFTMAX_ACCURATE)
		.value("CUDNN_SOFTMAX_LOG",
				CUDNN_SOFTMAX_LOG);

	py::enum_<SoftmaxMode>(m, "SoftmaxMode")
		.value("CUDNN_SOFTMAX_MODE_INSTANCE",
				CUDNN_SOFTMAX_MODE_INSTANCE)
		.value("CUDNN_SOFTMAX_MODE_CHANNEL",
				CUDNN_SOFTMAX_MODE_CHANNEL);

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
				CUDNN_STATUS_LICENSE_ERROR);

	py::enum_<TensorFormat>(m, "TensorFormat")
		.value("CUDNN_TENSOR_NCHW",
				CUDNN_TENSOR_NCHW)
		.value("CUDNN_TENSOR_NHWC",
				CUDNN_TENSOR_NHWC);

  m.def("get_version", []() {
    return cudnnGetVersion();
  });

  m.def("get_error_string", [](const Status& status) {
    return cudnnGetErrorString(status);
  });

  // m.def("cudnnCreate", ...) - Use Handle instead
  // TODO: throw exception saying exactly that

  // m.def("cudnnDestroy", ...) - Use Handle instead
  // TODO: throw exception saying exactly that

  // TODO: implement this
  //
  // Note: there's currently an compile issue with cudaStream_t type
  // m.def("set_stream", [](const Handle& handle, cudaStream_t streamId) {
  //   checkStatus(cudnnSetStream(handle, streamId));
  // });

  // m.def("get_stream", [](const Handle& handle) {
  //   cudaStream_t streamId;
  //   checkStatus(cudnnGetStream(handle, &streamId));
  //   return streamId;
  // });

  // m.def("cudnnCreateTensorDescriptor", ...) - Use TensorDescriptor instead
  // TODO: throw exception saying exactly that

  m.def("set_tensor_4d_descriptor",
    []( TensorDescriptor& tensorDesc,
        TensorFormat tensorFormat,
        DataType dataType,
        std::tuple<int, int, int, int> tensorDims) {
      int n = std::get<0>(tensorDims);
      int c = std::get<1>(tensorDims);
      int h = std::get<2>(tensorDims);
      int w = std::get<3>(tensorDims);
      checkStatus(
        cudnnSetTensor4dDescriptor(tensorDesc, tensorFormat, dataType, n, c, h, w)
      );
    });

  m.def("set_tensor_4d_descriptor_ex",
    []( TensorDescriptor& tensorDesc,
        DataType dataType,
        std::tuple<int, int, int, int> tensorDims,
        std::tuple<int, int, int, int> tensorStrides ) {

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
    });

  m.def("get_tensor_4d_descriptor",
    []( TensorDescriptor& tensorDesc ) {
      cudnnDataType_t dataType;

      int n;
      int c;
      int h;
      int w;

      int nS;
      int cS;
      int hS;
      int wS;

      checkStatus(
        cudnnGetTensor4dDescriptor(tensorDesc, &dataType, &n, &c, &h, &w, &nS, &cS, &hS, &wS)
      );

      return std::make_tuple(dataType, std::vector<int>({n, c, h, w}), std::vector<int>({nS, cS, hS, wS}));
    });

  m.def("set_tensor_nd_descriptor",
    []( TensorDescriptor& tensorDesc,
        DataType dataType,
        std::vector<int> dims,
        std::vector<int> strides ) {

      if (dims.size() != strides.size())
        throw std::length_error("dims and strides must be of the same length");

      checkStatus(
        cudnnSetTensorNdDescriptor(tensorDesc, dataType, dims.size(), dims.data(), strides.data())
      );
    });

  m.def("get_tensor_nd_descriptor",
    []( TensorDescriptor& tensorDesc,
        int numDimsRequested = 1) {

      cudnnDataType_t dataType;
      int numDims;

      std::vector<int> dims, strides;
      for (int i = 0; i < numDimsRequested; i++) {
        dims.push_back(0);
        strides.push_back(0);
      }

      checkStatus(
        cudnnGetTensorNdDescriptor(tensorDesc, numDimsRequested, &dataType, &numDims, dims.data(), strides.data())
      );

      return std::make_tuple(dataType, dims, strides);
    });

  // m.def("cudnnDestroyTensorDescriptor", ...) - Use TensorDescriptor instead
  // TODO: throw exception saying exactly that


  // cudnnStatus_t
  // cudnnTransformTensor( cudnnHandle_t                  handle,
  //                       const void                    *alpha,
  //                       const cudnnTensorDescriptor_t  xDesc,
  //                       const void                    *x,
  //                       const void                    *beta,
  //                       const cudnnTensorDescriptor_t  yDesc,
  //                       void                          *y )

  // cudnnStatus_t
  // cudnnAddTensor_(  cudnnHandle_t handle,
  //                   const void *alpha,
  //                   const cudnnTensorDescriptor_t aDesc,
  //                   const void *A,
  //                   const void *beta,
  //                   const cudnnTensorDescriptor_t cDesc,
  //                   void *C)

	return m.ptr();
}
