#ifndef PYCUDNN_TENSOR_DESCRIPTOR_HPP
#define PYCUDNN_TENSOR_DESCRIPTOR_HPP

#include <cudnn.h>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {
    struct TensorDescriptor :
      RAII< cudnnTensorDescriptor_t,
            cudnnCreateTensorDescriptor,
            cudnnDestroyTensorDescriptor > {};
}

#endif // PYCUDNN_TENSOR_DESCRIPTOR_HPP
