#ifndef PYCUDNN_POOLING_DESCRIPTOR_HPP
#define PYCUDNN_POOLING_DESCRIPTOR_HPP

#include <cudnn.h>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {
    class PoolingDescriptor :
    public RAII< cudnnPoolingDescriptor_t,
                    cudnnCreatePoolingDescriptor,
                    cudnnDestroyPoolingDescriptor > {};
} // PyCuDNN

#endif // PYCUDNN_POOLING_DESCRIPTOR_HPP
