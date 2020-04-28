#ifndef CUCCL_LE_CUH
#define CUCCL_LE_CUH

#include <cuda_runtime.h>
//#ifndef TYPE
#define TYPE int
//#endif // !TYPE



namespace CUCCL{

__device__ int getMinor(int a, int b);

__device__ TYPE getDiff(TYPE a, TYPE b);

__global__ void InitCCL(int labelList[], int reference[], int width, int height);

__global__ void scanning(TYPE frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height, TYPE threshold);

__global__ void scanning8(TYPE frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height, TYPE threshold);

__global__ void analysis(int labelList[], int reference[], int width, int height);

__global__ void labelling(int labelList[], int reference[], int width, int height);

class CCLLEGPU
{
public:
	explicit CCLLEGPU(TYPE* dataOnDevice = nullptr, int* labelListOnDevice = nullptr, int* referenceOnDevice = nullptr)
		: FrameDataOnDevice(dataOnDevice),
		  LabelListOnDevice(labelListOnDevice),
		  ReferenceOnDevice(referenceOnDevice)
	{
	}

	void CudaCCL(TYPE* frame, int* labels, int width, int height, int degreeOfConnectivity, TYPE threshold);

private:
	TYPE* FrameDataOnDevice;
	int* LabelListOnDevice;
	int* ReferenceOnDevice;
};

}

#endif