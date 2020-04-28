
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <stdio.h>
#include "my_functions.h"

#define INF 9999999999
#define PLATEAU 0
#define BLOCK_SIZE 6
// Convert 2D index to 1D index.
#define INDEX(j,i,ld) ((j) * ld + (i))

// Convert local (shared memory) coord to global (image) coordinate.
#define L2I(ind,off) (((ind) / BLOCK_SIZE) * (BLOCK_SIZE - 2)-1+(off))

// Texture memory for image.
texture<float, 2> img;

// Neighbour pixel generator (N-W to W order).
__constant__ int N_xs[8] = { -1,0,1,1,1,0,-1,-1 };
__constant__ int N_ys[8] = { -1,-1,-1,0,1,1,1,0 };

// Step 1.
__global__ void descent_kernel(float* labeled, const int w, const int h)
{
	int tx = threadIdx.x;  int ty = threadIdx.y;
	int bx = blockIdx.x;   int by = blockIdx.y;
	int bdx = blockDim.x;  int bdy = blockDim.y;
	int i = bdx * bx + tx; int j = bdy * by + ty;

	__shared__ float s_I[BLOCK_SIZE*BLOCK_SIZE];
	int size = BLOCK_SIZE - 2;
	int img_x = L2I(i, tx);
	int img_y = L2I(j, ty);
	int new_w = w + w * 2;
	int new_h = h + h * 2;
	int p = INDEX(img_y, img_x, w);

	int ghost = (tx == 0 || ty == 0 ||
		tx == bdx - 1 || ty == bdy - 1);

	if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
		(bx == (w / size - 1) && tx == bdx - 1) ||
		(by == (h / size - 1) && ty == bdy - 1)) {
		s_I[INDEX(ty, tx, BLOCK_SIZE)] = INF;
	}
	else {
		s_I[INDEX(ty, tx, BLOCK_SIZE)] = tex2D(img, img_x, img_y);
	}

	__syncthreads();

	if (j < new_h && i < new_w && ghost == 0) {
		float I_q_min = INF;
		float I_p = tex2D(img, img_x, img_y);

		int exists_q = 0;

		for (int k = 0; k < 8; k++) {
			int n_x = N_xs[k] + tx; int n_y = N_ys[k] + ty;
			float I_q = s_I[INDEX(n_y, n_x, BLOCK_SIZE)];
			if (I_q < I_q_min) I_q_min = I_q;
		}

		for (int k = 0; k < 8; k++) {
			int x = N_xs[k]; int y = N_ys[k];
			int n_x = x + tx; int n_y = y + ty;
			int n_tx = L2I(i, n_x); int n_ty = L2I(j, n_y);
			float I_q = s_I[INDEX(n_y, n_x, BLOCK_SIZE)];
			int q = INDEX(n_ty, n_tx, w);
			if (I_q < I_p && I_q == I_q_min) {
				labeled[p] = -q;
				exists_q = 1; break;
			}
		}
		if (exists_q == 0) labeled[p] = PLATEAU;
	}

}

// Step 2A.
__global__ void increment_kernel(float* L, const int w, const int h)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int p = INDEX(j, i, w);

	if (j < h && i < w && L[p] == PLATEAU) {
		L[p] = p + 1;
	}
}

// Step 2B.
__global__ void minima_kernel(float* L, int* C, const int w, const int h)
{
	int tx = threadIdx.x;  int ty = threadIdx.y;
	int bx = blockIdx.x;   int by = blockIdx.y;
	int bdx = blockDim.x;  int bdy = blockDim.y;
	int i = bdx * bx + tx; int j = bdy * by + ty;

	__shared__ float s_L[BLOCK_SIZE*BLOCK_SIZE];
	int size = BLOCK_SIZE - 2;
	int img_x = L2I(i, tx);
	int img_y = L2I(j, ty);
	int true_p = INDEX(img_y, img_x, w);
	int s_p = INDEX(ty, tx, BLOCK_SIZE);
	int new_w = w + w * 2;
	int new_h = h + h * 2;
	int ghost = (tx == 0 || ty == 0 ||
		tx == bdx - 1 || ty == bdy - 1) ? 1 : 0;

	if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
		(bx == (w / size - 1) && tx == bdx - 1) ||
		(by == (h / size - 1) && ty == bdy - 1)) {
		s_L[INDEX(ty, tx, BLOCK_SIZE)] = INF;
	}
	else {
		s_L[s_p] = L[INDEX(img_y, img_x, w)];
	}

	__syncthreads();

	int active = (j < new_h && i <
		new_w && s_L[s_p] > 0) ? 1 : 0;

	if (active == 1 && ghost == 0) {
		for (int k = 0; k < 8; k++) {
			int n_x = N_xs[k] + tx; int n_y = N_ys[k] + ty;
			int s_q = INDEX(n_y, n_x, BLOCK_SIZE);
			if (s_L[s_q] == INF) continue;
			if (s_L[s_q] > s_L[s_p])
				s_L[s_p] = s_L[s_q];
		}
		if (L[true_p] != s_L[s_p]) {
			L[true_p] = s_L[s_p];
			atomicAdd(&C[0], 1);
		}
	}
}


// Step 3.
__global__ void plateau_kernel(float* L, int* C, const int w, const int h)
{
	int tx = threadIdx.x;  int ty = threadIdx.y;
	int bx = blockIdx.x;   int by = blockIdx.y;
	int bdx = blockDim.x;  int bdy = blockDim.y;
	int i = bdx * bx + tx; int j = bdy * by + ty;

	__shared__ float s_L[BLOCK_SIZE*BLOCK_SIZE];
	int size = BLOCK_SIZE - 2;
	int img_x = L2I(i, tx);
	int img_y = L2I(j, ty);
	int true_p = INDEX(img_y, img_x, w);
	int p = INDEX(ty, tx, BLOCK_SIZE);
	int new_w = w + w * 2;
	int new_h = h + h * 2;
	int ghost = (tx == 0 || ty == 0 ||
		tx == bdx - 1 || ty == bdy - 1);

	// Load data into shared memory.
	if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
		(bx == (w / size - 1) && tx == bdx - 1) ||
		(by == (h / size - 1) && ty == bdy - 1)) {
		s_L[INDEX(ty, tx, BLOCK_SIZE)] = INF;
	}
	else {
		s_L[INDEX(ty, tx, BLOCK_SIZE)] =
			L[INDEX(img_y, img_x, w)];
	}

	__syncthreads();

	if (j < new_h && i < new_w &&
		s_L[p] == PLATEAU && ghost == 0) {
		float I_p = tex2D(img, img_x, img_y);
		float I_q;
		int n_x, n_y; float L_q;

		for (int k = 0; k < 8; k++) {
			n_x = N_xs[k] + tx; n_y = N_ys[k] + ty;
			L_q = s_L[INDEX(n_y, n_x, BLOCK_SIZE)];
			if (L_q == INF || L_q >= 0) continue;
			int n_tx = L2I(i, n_x); int n_ty = L2I(j, n_y);
			int q = INDEX(n_ty, n_tx, w);
			I_q = tex2D(img, n_tx, n_ty);
			if (I_q == I_p && L[true_p] != -q) {
				L[true_p] = -q;
				atomicAdd(&C[0], 1);
				break;
			}
		}
	}

}
// Step 4.
__global__ void flood_kernel(float* L, int* C, const int w, const int h)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int p = INDEX(j, i, w); int q;

	if (j < h && i < w && L[p] <= 0) {
		q = -L[p];
		if (L[q] > 0 && L[p] != L[q]) {
			L[p] = L[q];
			atomicAdd(&C[0], 1);
		}
	}
}


void watershed(float *image_map, float *labels, int width, int height) {
	dim3 block13(6, 6, 1);
	dim3 grid13(width / (block13.x - 2), height / (block13.y - 2));

	dim3 block4(16, 16, 1);
	dim3 grid4(width / (block4.x - 2), height / (block4.y - 2));

	memset(labels, 0, width * height);

	int count = 0;

	float *dev_labeled;
	size_t dev_labeled_pitch;
	cudaMalloc((void**)&dev_labeled, width * sizeof(float) * height);
	cudaMemcpy(dev_labeled, labels, width * sizeof(float) * height, cudaMemcpyHostToDevice);

	int *dev_count;
	cudaMalloc((void**)&dev_count, sizeof(int));
	cudaMemcpy(&dev_count, &count, sizeof(int), cudaMemcpyHostToDevice);

	float *dev_image;
	size_t dev_image_pitch;
	cudaMallocPitch((void**)&dev_image, &dev_image_pitch, width * sizeof(float), height);
	cudaMemcpy2D(dev_image, dev_image_pitch, image_map, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
	cudaBindTexture2D(0, img, dev_image, cudaCreateChannelDesc<float>(), width, height, dev_image_pitch);

	{
		Timer stopwatch("Kernels only");
		//Step 1
		descent_kernel << <grid13, block13 >> > (dev_labeled, width, height);

		//Step 2
		increment_kernel << <grid13, block13 >> > (dev_labeled, width, height);

		int old = -1;
		int new_ = -2;

		while (old != new_) {
			old = new_;
			minima_kernel << <grid13, block13 >> > (dev_labeled, dev_count, width, height);
			cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
			new_ = count;
		}

		//Step 3
		cudaMemset(dev_count, 0, sizeof(int));
		old = -1;
		new_ = -2;
		while (old != new_)
		{
			old = new_;
			plateau_kernel << <grid13, block13 >> > (dev_labeled, dev_count, width, height);
			cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
			new_ = count;
		}

		//Step 4
		cudaMemset(dev_count, 0, sizeof(int));
		old = -1;
		new_ = -2;
		while (old != new_)
		{
			old = new_;
			flood_kernel << <grid4, block4 >> > (dev_labeled, dev_count, width, height);
			cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);
			new_ = count;
		}

		cudaDeviceSynchronize();
	}

	cudaMemcpy(labels, dev_labeled, width * height * sizeof(float), cudaMemcpyDeviceToHost);
	cudaUnbindTexture(img);
	cudaFree(dev_count);
	cudaFree(dev_image);
	cudaFree(dev_labeled);
}


int main(int argc, char** args)
{
	float *image_map;
	int height, width;
	float *labels;

	load_image(args[1], &image_map, &width, &height);
	labels = new float[height * width];

	cudaFree(nullptr);
	{
		Timer x("Watershed in total took: ");
		watershed(image_map, labels, width, height);
	}

	save_labels(labels, width, height, args[2]);

	delete[] labels;
	delete[] image_map;

    return 0;
}