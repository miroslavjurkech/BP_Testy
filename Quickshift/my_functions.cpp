#include "my_functions.h"

#define cimg_use_png
#include "../CImg/CImg.h"

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include "quickshift_common.h"

//#define PRINT_BASE_RANGE
//#define GRAYSCALE_CHECK

Timer::Timer(std::string timer_for)
	:m_timer_for(timer_for)
{
	m_start = std::chrono::system_clock::now();
}

Timer::~Timer()
{
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	std::cout << m_timer_for << " took " << std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count() << " us.\n";
}


int count_segments(int *frame, int width, int height) {

	std::vector<bool> bitmap;
	bitmap.resize(width * height, false);

	int total_len = width * height;

	for (int i = 0; i < total_len; i++) {
		bitmap[frame[i]] = true;
	}

	int total_sum_of_segments = 0;
	for (int i = 0; i < total_len; i++) {
		if (bitmap[i]) total_sum_of_segments++;
	}

	return total_sum_of_segments;
}

void save_labels(int *labels, int width, int height, std::string filename) {
	cimg_library::CImg<unsigned char> dest(width, height, 1, 3);

	for (int i = 0; i < height; i++) {
		for (int e = 0; e < width; e++) {
			unsigned char colors[3];
			colors[0] = (unsigned char)labels[i * width + e];
			labels[i * width + e] >>= 4;
			colors[1] = (unsigned char)labels[i * width + e];
			labels[i * width + e] >>= 4;
			colors[2] = (unsigned char)labels[i * width + e];

			dest.draw_point(e, i, colors);
		}
	}

	dest.save(filename.c_str());

	std::cout << "Labels saved as " << filename << ".\n";
}

void load_image(std::string filename, float **frame, int *width, int *height) {
	cimg_library::CImg<unsigned short> src(filename.c_str());
	*width = src.width();
	*height = src.height();

#ifdef GRAYSCALE_CHECK
	for (int i = 0; i < src.height(); i++) {  //divide by max - normalization
		for (int e = 0; e < src.width(); e++) {
			if ((*src.data(e, i, 0, 0)) != (*src.data(e, i, 0, 1)) || (*src.data(e, i, 0, 0)) != (*src.data(e, i, 0, 2))) {
				std::cerr << "Input error: Not a grayscale image!\n";
				exit(EXIT_FAILURE);
			}
		}
	}
#endif // GRAYSCALE_CHECK


	*frame = new float[src.width() * src.height()];

	if (*frame == NULL) {
		std::cerr << "Error allocating memory\n";
		abort();
	}

	float max_val = *(src.data());
	float min_val = *(src.data());
	for (int i = 0; i < src.height(); i++) {  //Init frame and find max //opt min
		for (int e = 0; e < src.width(); e++) {
			*((*frame) + i * src.width() + e) = (float)*(src.data() + i * src.width() + e);
			if (*((*frame) + i * src.width() + e) > max_val) {
				max_val = *((*frame) + i * src.width() + e);
			}
			if (*((*frame) + i * src.width() + e) < min_val) {
				min_val = *((*frame) + i * src.width() + e);
			}
		}
	}
#ifdef PRINT_BASE_RANGE
	std::cout << "Base values in range from " << min_val << " to " << max_val << ".\n";
#endif // PRINT_BASE_RANGE


	for (int i = 0; i < src.height(); i++) {  //divide by max - normalization
		for (int e = 0; e < src.width(); e++) {
			*((*frame) + i * src.width() + e) /= max_val;
		}
	}

	float *xx = new float[src.width() * src.height()];

	for (int i = 0; i < src.height(); i++) {
		for (int e = 0; e < src.width(); e++)
			xx[e * src.height() + i] = *((*frame) + i * src.width() + e);
	}

	delete[] (*frame);
	*frame = xx;

	std::cout << filename << " loaded. Height: " << *height << " Width: " << *width << std::endl;
}

void cimg_to_matlab(const std::string& name, image_t& im) {
	float *frame;
	int height, width;

	load_image(name, &frame, &width, &height);

	im.N1 = height;
	im.N2 = width;
	im.K = 1;
	//im.I = (float *)calloc(im.N1*im.N2*im.K, sizeof(float));
	im.I = frame;
	for (int k = 0; k < im.K; k++) //k = 1 
		for (int col = 0; col < im.N2; col++)
			for (int row = 0; row < im.N1; row++)
			{
				//unsigned char * pt = IMG.getPixelPt(col, im.N1 - 1 - row);
				im.I[row + col * im.N1 + k * im.N1*im.N2] *= 32.0;
				//im.I[row + col * im.N1 + k * im.N1*im.N2] /= 255.0;// Scale 0-32
			}

	float *frame2 = new float[im.N1 * im.N2 * 3];

	memcpy(frame2, frame, im.N1 * im.N2 * sizeof(float));
	memcpy(frame2 + im.N1 * im.N2, frame, im.N1 * im.N2 * sizeof(float));
	memcpy(frame2 + 2 * im.N1 * im.N2, frame, im.N1 * im.N2 * sizeof(float));

	im.I = frame2;
	delete[] frame;
	im.K = 3;
}