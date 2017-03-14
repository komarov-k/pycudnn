#ifndef PYCUDNN_RAII_HPP
#define PYCUDNN_RAII_HPP

#include <memory> // std::shared_ptr

#include "PyCuDNNStatus.hpp" // checkStatus

namespace PyCuDNN {

	template <typename T, Status (*C)(T*), Status (*D)(T)>
	class RAII {

		struct Resource {
			T object;

			Resource() {
				checkStatus(C(&object));
			}

			~Resource() {
				checkStatus(D(object));
			}
		};

  		std::shared_ptr<Resource> mResource;

  public:

  	RAII() : mResource(new Resource) {}

		T get() const {
			return mResource->object;
		}

    operator T() const {
      return mResource->object;
    }
	};
}

#endif // PYCUDNN_RAII_HPP
