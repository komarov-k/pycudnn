#ifndef PYCUDNN_ACTIVATION_DESCRIPTOR_HPP
#define PYCUDNN_ACTIVATION_DESCRIPTOR_HPP

#include <cudnn.h>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {
    class ActivationDescriptor :
        public RAII< cudnnActivationDescriptor_t,
                    cudnnCreateActivationDescriptor,
                    cudnnDestroyActivationDescriptor > {};
}

#endif // PYCUDNN_ACTIVATION_DESCRIPTOR_HPP
