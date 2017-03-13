#ifndef PYCUDNN_DATA_TYPE_HPP
#define PYCUDNN_DATA_TYPE_HPP

#include <cudnn.h>
#include <cstdint>

namespace PyCuDNN {

  typedef cudnnDataType_t DataType;

  template <typename T> struct dataType {};

  /**
   * @brief      Class for data type.
   */
  template<>
      class dataType<float> {
          static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
      };

  /**
   * @brief      Class for data type.
   */
  template <>
      class dataType<double> {
          static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
      };
}

#endif // PYCUDNN_DATA_TYPE
