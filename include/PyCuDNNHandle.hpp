#ifndef PYCUDNN_HANDLE
#define PYCUDNN_HANDLE

#include <cudnn.h>

#include "PyCuDNNRAII.hpp" // RAII

namespace PyCuDNN {

    class Handle :
        public RAII< cudnnHandle_t,
                    cudnnCreate,
                    cudnnDestroy > {};

    // /**
    //  * @brief      Wrapper class for cuDNN library handle.
    //  */
    // class Handle {

    //  /**
    //   * @brief      RAII wrapper around C-struct handle.
    //   */
    //  struct Wrapped {
    //      cudnnHandle_t object;

    //      Wrapped() {
    //          checkStatus(cudnnCreate(&object));
    //      }

    //      ~Wrapped() {
 //                checkStatus(cudnnDestroy(object));
    //      }
    //  };

    //  /**
    //   * Pointer to cuDNN handle wrapper.
    //   */
    //  std::shared_ptr<Wrapped> mWrapped;

 //    public:

 //     /**
 //      * @brief      Constructs new Handle instance.
 //      *
 //      * @param[in]  deviceId  NVIDIA GPU device identifier
 //      */
 //     Handle() : mWrapped(new Wrapped()) {}

 //     /**
 //      * @brief      Accessor for underlying (C-struct) cuDNN handle object.
 //      *
 //      * @return     cuDNN handle object.
 //      */
 //     cudnnHandle_t get() const {
 //         return mWrapped->object;
 //     }
    // };
}

#endif // PYCUDNN_HANDLE
