#ifndef PYCUDNN_LRN_DESCRIPTOR_HPP
#define PYCUDNN_LRN_DESCRIPTOR_HPP

#include <cudnn.h>

#include "PyCuDNNRAII.hpp"

namespace PyCuDNN {

  class LRNDescriptor :
    RAII< cudnnLRNDescriptor_t,
          cudnnCreateLRNDescriptor,
          cudnnDestroyLRNDescriptor > {};

} // PyCuDNN
#endif
