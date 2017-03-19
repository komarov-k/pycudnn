#ifndef PYCUDNN_STATUS_HPP
#define PYCUDNN_STATUS_HPP

#include <cudnn.h>

namespace PyCuDNN {
	typedef cudnnStatus_t Status;

    class Exception {
      Status mStatus;
    public:
      Exception(Status status) : mStatus(status) {}
       
      const char* what() const noexcept {
        switch (mStatus) {
          case CUDNN_STATUS_SUCCESS:
            return "PyCuDNN.Exception: CUDNN_STATUS_SUCCESS";
          case CUDNN_STATUS_NOT_INITIALIZED:
            return "PyCuDNN.Exception: CUDNN_STATUS_NOT_INITIALIZED";
          case CUDNN_STATUS_ALLOC_FAILED:
            return "PyCuDNN.Exception: CUDNN_STATUS_ALLOC_FAILED";
          case CUDNN_STATUS_BAD_PARAM:
            return "PyCuDNN.Exception: CUDNN_STATUS_BAD_PARAM";
          case CUDNN_STATUS_ARCH_MISMATCH:
            return "PyCuDNN.Exception: CUDNN_STATUS_ARCH_MISMATCH";
          case CUDNN_STATUS_MAPPING_ERROR:
            return "PyCuDNN.Exception: CUDNN_STATUS_MAPPING_ERROR";
          case CUDNN_STATUS_EXECUTION_FAILED:
            return "PyCuDNN.Exception: CUDNN_STATUS_EXECUTION_FAILED";
          case CUDNN_STATUS_INTERNAL_ERROR:
            return "PyCuDNN.Exception: CUDNN_STATUS_INTERNAL_ERROR";
          case CUDNN_STATUS_NOT_SUPPORTED:
            return "PyCuDNN.Exception: CUDNN_STATUS_NOT_SUPPORTED";
          case CUDNN_STATUS_LICENSE_ERROR:
            return "PyCuDNN.Exception: CUDNN_STATUS_LICENSE_ERROR";
          default:
            return "PyCuDNN.Exception: CUDNN_STATUS_UNKNOWN";
        };
      };

      Status getStatus() {
        return mStatus;
      }
    };

	void checkStatus(Status status) {
      if (status != CUDNN_STATUS_SUCCESS) {
        throw Exception(status);
      }
	}
}

#endif // PYCUDNN_STATUS_HPP
