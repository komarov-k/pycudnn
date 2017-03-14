#ifndef PYCUDNN_DROPOUT_DESCRIPTOR_HPP
#define PYCUDNN_DROPOUT_DESCRIPTOR_HPP

#include <cudnn.h>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {
    class DropoutDescriptor :
        public RAII< cudnnDropoutDescriptor_t,
                    cudnnCreateDropoutDescriptor,
                    cudnnDestroyDropoutDescriptor > {};
} // PyCuDNN

#endif // PYCUDNN_DROPOUT_DESCRIPTOR_HPP
