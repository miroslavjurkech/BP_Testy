#ifndef CUCCL_NP_CUH
#define CUCCL_NP_CUH

#include <host_defines.h>

namespace CUCCL{

__device__ int atom_MIN(int a, int b);

__global__ void InitCCL(int labelList[], int width, int height);

__global__ void kernel4(int dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold);

__global__ void kernel8(int dataOnDevice[], int labelOnDevice[], bool* markFlagOnDevice, int N, int width, int height, int threshold);

class CCLNPGPU
{
public:
	explicit CCLNPGPU(int* dataOnDevice = nullptr, int* labelListOnDevice = nullptr)
		: FrameDataOnDevice(dataOnDevice),
		  LabelListOnDevice(labelListOnDevice)
	{
	}

	void CudaCCL(int* frame, int* labels, int width, int height, int degreeOfConnectivity, int threshold);

private:
	int* FrameDataOnDevice;
	int* LabelListOnDevice;
};


}
#endif



