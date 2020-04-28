#include "my_functions.h"

#define cimg_use_png
#include "../CImg/CImg.h"

#include "../glm/glm.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <chrono>

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

	int max_label = labels[0];
	for (int i = 0; i < height; i++) {
		for (int e = 0; e < width; e++) {
			if (max_label < labels[i * width + e]) {
				max_label = labels[i * width + e];
			}
		}
	}

	for (int i = 0; i < height; i++) {
		for (int e = 0; e < width; e++) {
			unsigned char colors[3];
			//float sigma = (float)labels[i * width + e] / (width * height);
			//float sigma = (float)labels[i * width + e] / INT_MAX;
			float sigma = (float)labels[i * width + e] / max_label;
			float r = glm::fract(std::sin(sigma + 1.0f) * 43758.5453f);
			float g = glm::fract(std::sin(sigma + 3.59f) * 43758.5453f);
			float b = glm::fract(std::sin(sigma + 6.7f) * 43758.5453f);
			colors[0] = (unsigned char)(r * 255);
			colors[1] = (unsigned char)(g * 255);
			colors[2] = (unsigned char)(b * 255);
			dest.draw_point(e, i, colors);
		}
	}

	dest.save(filename.c_str());

	std::cout << "Labels saved as " << filename << ".\n";
}

void load_image(std::string filename, int **frame, int *width, int *height) {
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


	*frame = new int[src.width() * src.height()];

	if (*frame == NULL) {
		std::cerr << "Error allocating memory\n";
		abort();
	}

	int max_val = *(src.data());
	int min_val = *(src.data());
#ifdef PRINT_BASE_RANGE
#endif // PRINT_BASE_RANGE
	for (int i = 0; i < src.height(); i++) {  //Init frame and find max //opt min
		for (int e = 0; e < src.width(); e++) {
			*((*frame) + i * src.width() + e) = *(src.data() + i * src.width() + e);
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


/*	for (int i = 0; i < src.height(); i++) {  //divide by max - normalization
		for (int e = 0; e < src.width(); e++) {
			*((*frame) + i * src.width() + e) /= max_val;
		}
	}*/

	std::cout << filename << " loaded. Height: " << *height << " Width: " << *width << std::endl;
}