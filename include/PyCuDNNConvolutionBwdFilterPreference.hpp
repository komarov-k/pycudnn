#ifndef PYCUDNN_CONVOLUTION_BWD_FILTER_PREFERENCE_HPP
#define PYCUDNN_CONVOLUTION_BWD_FILTER_PREFERENCE_HPP

#include <cudnn.h>

#if CUDNN_VERSION < 8000
namespace PyCuDNN {
	typedef cudnnConvolutionBwdFilterPreference_t ConvolutionBwdFilterPreference;
} // PyCuDNN
#endif // CUDNN_VERSION

#endif // PYCUDNN_CONVOLUTION_BWD_FILTER_PREFERENCE_HPP
