#ifndef PYCUDNN_CONVOLUTION_DESCRIPTOR_HPP
#define PYCUDNN_CONVOLUTION_DESCRIPTOR_HPP

#include <cudnn.h>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {
    class ConvolutionDescriptor :
      public RAII<  cudnnConvolutionDescriptor_t,
                    cudnnCreateConvolutionDescriptor,
                    cudnnDestroyConvolutionDescriptor > {};
}

#endif // PYCUDNN_CONVOLUTION_DESCRIPTOR_HPP
