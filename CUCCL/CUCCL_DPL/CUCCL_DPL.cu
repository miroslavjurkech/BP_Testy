#ifndef CUCCL_DPL_CU
#define CUCCL_DPL_CU

#include <host_defines.h>
#include "CUCCL_DPL.cuh"
#include "..\CUCCL_NP\CUCCL_NP.cuh"
#include <device_launch_parameters.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>

#include "../my_functions.h"


namespace CUCCL{
	const int BLOCK = 8;

// catch CUDA error
#define CHECK(call)                                                            \
{                                                                              \
	const cudaError_t error = call;                                            \
	if (error != cudaSuccess)                                                  \
	{                                                                          \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
		fprintf(stderr, "code: %d, reason: %s\n", error,                       \
			cudaGetErrorString(error));                                    \
		exit(1);                                                               \
	}                                                                          \
}

	__global__ void InitCCL2(int L_d[], int width, int height)
	{
		// interger Ld[N] N = width * height

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height)
			return;

		int id = x + y * width;

		L_d[id] = id;
	}


__global__
void DPL_CCL_UD(const int width, const int height, int* labels, const size_t lpitch, int* edges, const size_t edpitch, char* changed, const int threshold) {

	size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_id >= width)
		return;

	char _changed = 0;

	int my_label = labels[thread_id];
	for (int row = 1; row < height; ++row) {
		int neighbour_label = labels[row * lpitch + thread_id];
		//merging occures
		if ( abs(edges[(row - 1) * edpitch + thread_id] - edges[row * edpitch + thread_id]) <= threshold && my_label != neighbour_label) {
			my_label = min(my_label, neighbour_label);
			labels[row * lpitch + thread_id] = my_label;
			_changed = 1;
		}
		else {
			my_label = neighbour_label;
		}
	}

	if (_changed)
		*changed = 1;
}

__global__
void DPL_CCL_DU(const int width, const int height, int* labels, const size_t lpitch, int* edges, const size_t edpitch, char *changed, const int threshold) {

	size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_id >= width)
		return;

	char _changed = 0;

	int my_label = labels[thread_id + (height - 1) * lpitch];
	for (int row = height - 2; row >= 0; --row) {
		int neighbour_label = labels[row * lpitch + thread_id];
		//merging occures
		if ( abs(edges[(row + 1) * edpitch + thread_id] - edges[row * edpitch + thread_id]) <= threshold && my_label != neighbour_label) {
			my_label = min(my_label, neighbour_label);
			labels[row * lpitch + thread_id] = my_label;
			_changed = 1;
		}
		else {
			my_label = neighbour_label;
		}
	}

	if (_changed)
		*changed = 1;
}


__global__
void DPL_CCL_LR(const int width, const int height, int* labels, const size_t lpitch, int* edges, const size_t edpitch, char* changed, const int threshold) {

	size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_id >= height)
		return;

	char _changed = 0;

	//int my_label = labels[thread_id + (height - 1) * lpitch];
	int my_label = labels[thread_id * lpitch];
	for (int column = 1; column < width; ++column) {
		int neighbour_label = labels[thread_id * lpitch + column];
		//merging occures
		if ( abs(edges[thread_id * edpitch + column] - edges[thread_id * edpitch + column - 1]) <= threshold && my_label != neighbour_label) {
			my_label = min(my_label, neighbour_label);
			labels[thread_id * lpitch + column] = my_label;
			_changed = 1;
		}
		else {
			my_label = neighbour_label;
		}
	}

	if (_changed)
		*changed = 1;
}

__global__
void DPL_CCL_RL(const int width, const int height, int* labels, const size_t lpitch, int* edges, const size_t edpitch, char* changed, const int threshold) {

	size_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread_id >= height)
		return;

	char _changed = 0;

	int my_label = labels[thread_id * lpitch + width - 1];
	for (int column = width - 2; column >= 0; --column) {
		int neighbour_label = labels[thread_id * lpitch + column];
		//merging occures
		if (abs(edges[thread_id * edpitch + column] - edges[thread_id * edpitch + column + 1]) <= threshold && my_label != neighbour_label) {
			my_label = min(my_label, neighbour_label);
			labels[thread_id * lpitch + column] = my_label;
			_changed = 1;
		}
		else {
			my_label = neighbour_label;
		}
	}

	if (_changed)
		*changed = 1;
}

__global__
void DIAGONAL_LU_RD(const int width, const int height, int* labels, const size_t lpitch, int* edges, const size_t edpitch, char* changed, const int threshold) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	id -= height - 1;
	if (id >= width) return;

	int my_row = 0;
	int my_column = id;

	char _changed = 0;
	for (int i = 0; i < (height - 1); ++i) {
		if (my_column >= 0 && my_column < (width - 1)) {
			int my_label = labels[my_row * lpitch + my_column];
			int neighbour_label = labels[(my_row + 1) * lpitch + my_column + 1];

			if (abs(edges[my_row * edpitch + my_column] - edges[(my_row + 1) * edpitch + my_column + 1]) <= threshold && my_label > neighbour_label) {
				labels[my_row * lpitch + my_column] = neighbour_label;
				_changed = 1;
			}
		}

		my_row++;
		my_column++;
	}

	if (_changed)
		*changed = 1;
}

__global__
void DIAGONAL_LD_RU(const int width, const int height, int* labels, const size_t lpitch, int* edges, const size_t edpitch, char* changed, const int threshold) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	id -= height - 1;
	if (id >= width) return;

	int my_row = height - 1;
	int my_column = id;

	char _changed = 0;
	for (int i = 0; i < (height - 1); ++i) {
		if (my_column >= 0 && my_column < (width - 1)) {
			int my_label = labels[my_row * lpitch + my_column];
			int neighbour_label = labels[(my_row - 1) * lpitch + my_column + 1];

			if (abs(edges[my_row * edpitch + my_column] - edges[(my_row - 1) * edpitch + my_column + 1]) <= threshold && my_label > neighbour_label) {
				labels[my_row * lpitch + my_column] = neighbour_label;
				_changed = 1;
			}
		}

		my_row--;
		my_column++;
	}

	if (_changed)
		*changed = 1;
}

__global__
void DIAGONAL_RU_LD(const int width, const int height, int* labels, const size_t lpitch, int* edges, const size_t edpitch, char* changed, const int threshold) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	//id -= height - 1;
	if (id >= (width + height - 1)) return;

	int my_row = 0;
	int my_column = id;

	char _changed = 0;
	for (int i = 0; i < (height - 1); ++i) {
		if (my_column > 0 && my_column < width) {
			int my_label = labels[my_row * lpitch + my_column];
			int neighbour_label = labels[(my_row + 1) * lpitch + my_column - 1];

			if (abs(edges[my_row * edpitch + my_column] - edges[(my_row + 1) * edpitch + my_column - 1]) <= threshold && my_label > neighbour_label) {
				labels[my_row * lpitch + my_column] = neighbour_label;
				_changed = 1;
			}
		}

		my_row++;
		my_column--;
	}

	if (_changed)
		*changed = 1;
}

__global__
void DIAGONAL_RD_LU(const int width, const int height, int* labels, const size_t lpitch, int* edges, const size_t edpitch, char* changed, const int threshold) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	//id -= height - 1;
	if (id >= (width + height - 1)) return;

	int my_row = height - 1;
	int my_column = id;

	char _changed = 0;
	for (int i = 0; i < (height - 1); ++i) {
		if (my_column > 0 && my_column < width) {
			int my_label = labels[my_row * lpitch + my_column];
			int neighbour_label = labels[(my_row - 1) * lpitch + my_column - 1];

			if (abs(edges[my_row * edpitch + my_column] - edges[(my_row - 1) * edpitch + my_column - 1]) <= threshold && my_label > neighbour_label) {
				labels[my_row * lpitch + my_column] = neighbour_label;
				_changed = 1;
			}
		}

		my_row--;
		my_column--;
	}

	if (_changed)
		*changed = 1;
}

cudaError_t dpl_4_mine(int* dev_result_map, int* dev_labels, int width, int height, int threshold, char* dev_changed) {
	dim3 labeling_block(32, 1);
	dim3 labeling_grid;
	labeling_grid.x = static_cast<unsigned int>(ceil(static_cast<double>(width) / labeling_block.x));
	labeling_grid.y = 1;

	dim3 labeling_block_side(32, 1);
	dim3 labeling_grid_side(static_cast<unsigned int>(ceil(static_cast<double>(height) / labeling_block.x)), 1);

	char changed = 1;
	while (changed)
	{
		cudaMemset(dev_changed, 0, sizeof(char));
		DPL_CCL_UD << <labeling_grid, labeling_block >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DPL_CCL_LR << <labeling_grid_side, labeling_block_side >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DPL_CCL_DU << <labeling_grid, labeling_block >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DPL_CCL_RL << <labeling_grid_side, labeling_block_side >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		cudaError_t error_code = cudaMemcpy(&changed, dev_changed, sizeof(char), cudaMemcpyDeviceToHost);
		if (error_code != cudaSuccess) return error_code;
	}

	return cudaSuccess;
}

cudaError_t dpl_8_mine(int* dev_result_map, int* dev_labels, int width, int height, int threshold, char* dev_changed) {
	dim3 labeling_block(32, 1);
	dim3 labeling_grid;
	labeling_grid.x = static_cast<unsigned int>(ceil(static_cast<double>(width) / labeling_block.x));
	labeling_grid.y = 1;

	dim3 labeling_block_side(32, 1);
	dim3 labeling_grid_side(static_cast<unsigned int>(ceil(static_cast<double>(height) / labeling_block.x)), 1);

	dim3 diagonal_block(32, 1);
	dim3 diagonal_grid(static_cast<unsigned int>(ceil(static_cast<double>(height + width) / labeling_block.x)), 1);

	char changed = 1;
	while (changed)
	{
		cudaMemset(dev_changed, 0, sizeof(char));
		DPL_CCL_UD << <labeling_grid, labeling_block >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DIAGONAL_RD_LU << <diagonal_grid, diagonal_block >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DPL_CCL_LR << <labeling_grid_side, labeling_block_side >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DIAGONAL_RU_LD << <diagonal_grid, diagonal_block >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DPL_CCL_DU << <labeling_grid, labeling_block >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DIAGONAL_LU_RD << <diagonal_grid, diagonal_block >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DPL_CCL_RL << <labeling_grid_side, labeling_block_side >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		DIAGONAL_LD_RU << <diagonal_grid, diagonal_block >> > (width, height, dev_labels, width, dev_result_map, width, dev_changed, threshold);
		cudaError_t error_code = cudaMemcpy(&changed, dev_changed, sizeof(char), cudaMemcpyDeviceToHost);
		if (error_code != cudaSuccess) return error_code;
	}

	return cudaSuccess;
}

void CCLDPLGPU::CudaCCL(int* frame, int* labels, int width, int height, int degreeOfConnectivity, int thre)
{
	// set the device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	//printf("> Starting at Device %d: %s\n\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	
	//std::cout << "CUDA DPL..." << std::endl;

	auto nSamples = width * height;

	// allocate data on device
	cudaMalloc((void**)&gData, sizeof(int) * nSamples);
	CHECK(cudaPeekAtLastError());
	cudaMalloc((void**)&gLabelList, sizeof(int) * nSamples);
	CHECK(cudaPeekAtLastError());

	cudaMemcpy(gData, frame, sizeof(int) * nSamples, cudaMemcpyHostToDevice);
	CHECK(cudaPeekAtLastError());

	bool* isChanged;
	cudaMalloc((void**)&isChanged, sizeof(bool));
	CHECK(cudaPeekAtLastError());

	// initialize the label list
	dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
	dim3 threads(BLOCK, BLOCK);

	cudaDeviceSynchronize();
	{
		Timer stopwatch("Init: ");
		InitCCL2 << <grid, threads >> > (gLabelList, width, height);
		cudaDeviceSynchronize();
	}
	// print out the initial labels
	//print_init_labels(width, height, labels);  <<----------------


	cudaDeviceSynchronize();
	bool flagHost = true;

	{
		Timer stopwatch("Kernels");

		
		if (degreeOfConnectivity == 8)
		{
			dpl_8_mine(gData, gLabelList, width, height, thre, (char*)isChanged);
		}
		else
		{
			dpl_4_mine(gData, gLabelList, width, height, thre, (char*)isChanged);
		}

		cudaDeviceSynchronize();
	}

	cudaMemcpy(&flagHost, isChanged, sizeof(bool), cudaMemcpyDeviceToHost);
	CHECK(cudaPeekAtLastError());

	// copy back the labeling results
	cudaMemcpy(labels, gLabelList, sizeof(int) * nSamples, cudaMemcpyDeviceToHost);
	CHECK(cudaPeekAtLastError());

	CHECK(cudaFree(gData));
	CHECK(cudaFree(gLabelList));

	CHECK(cudaDeviceReset());
}
}

#endif