#ifndef PYCUDNN_CONVOLUTION_FWD_PREFERENCE_HPP
#define PYCUDNN_CONVOLUTION_FWD_PREFERENCE_HPP

#include <cudnn.h>

#if CUDNN_VERSION < 8000
namespace PyCuDNN {
	typedef cudnnConvolutionFwdPreference_t ConvolutionFwdPreference;
} // PyCuDNN
#endif // CUDNN_VERSION

#endif // PYCUDNN_CONVOLUTION_FWD_PREFERENCE_HPP
