#ifndef PYCUDNN_SPATIAL_TRANSFORMER_DESCRIPTOR_HPP
#define PYCUDNN_SPATIAL_TRANSFORMER_DESCRIPTOR_HPP

#include <cudnn.h>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {
    class SpatialTransformerDescriptor :
        public RAII< cudnnSpatialTransformerDescriptor_t,
                    cudnnCreateSpatialTransformerDescriptor,
                    cudnnDestroySpatialTransformerDescriptor > {};
} // PyCuDNN

#endif // PYCUDNN_SPATIAL_TRANSFORMER_DESCRIPTOR_HPP
