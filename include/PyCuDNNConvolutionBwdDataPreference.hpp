#ifndef PYCUDNN_CONVOLUTION_BWD_DATA_PREFERENCE_HPP
#define PYCUDNN_CONVOLUTION_BWD_DATA_PREFERENCE_HPP

#include <cudnn.h>

#if CUDNN_VERSION < 8000
namespace PyCuDNN {
	typedef cudnnConvolutionBwdDataPreference_t ConvolutionBwdDataPreference;
} // PyCuDNN
#endif // CUDNN_VERSION

#endif // PYCUDNN_CONVOLUTION_BWD_DATA_PREFERENCE_HPP
