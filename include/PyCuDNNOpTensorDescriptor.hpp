#ifndef PYCUDNN_OP_TENSOR_DESCRIPTOR_HPP
#define PYCUDNN_OP_TENSOR_DESCRIPTOR_HPP

#include <cudnn.h>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {

    class OpTensorDescriptor :
        public RAII< cudnnOpTensorDescriptor_t,
                    cudnnCreateOpTensorDescriptor,
                    cudnnDestroyOpTensorDescriptor > {};
} // PyCuDNN

#endif // PYCUDNN_OP_TENSOR_DESCRIPTOR_HPP
