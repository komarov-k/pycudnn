#ifndef PYCUDNN_FILTER_DESCRIPTOR_HPP
#define PYCUDNN_FILTER_DESCRIPTOR_HPP

#include <cudnn.h>
#include <cstdint>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {
	class FilterDescriptor :
		protected RAII< cudnnFilterDescriptor_t,
                    cudnnCreateFilterDescriptor,
                    cudnnDestroyFilterDescriptor > {};
}

#endif // PYCUDNN_FILTER_DESCRIPTOR_HPP
