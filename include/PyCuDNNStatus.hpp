#ifndef PYCUDNN_STATUS_HPP
#define PYCUDNN_STATUS_HPP

#include <cudnn.h>

namespace PyCuDNN {
	typedef cudnnStatus_t Status;
	
	void checkStatus(Status status) {
		// TODO: implement this
	}
}

#endif // PYCUDNN_STATUS_HPP
