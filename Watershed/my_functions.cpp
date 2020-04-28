#include "my_functions.h"

//Requires CIMG and libpng
#define cimg_use_png
#include "../CImg/CImg.h"

#include "../glm/glm.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <chrono>


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


void save_labels(float *labels, int width, int height, std::string filename) {
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

void load_image(std::string filename, float **frame, int *width, int *height) {
	cimg_library::CImg<unsigned short> src(filename.c_str());
	*width = src.width();
	*height = src.height();


	*frame = new float[src.width() * src.height()];

	if (*frame == NULL) {
		std::cerr << "Error allocating memory\n";
		abort();
	}

	for (int i = 0; i < src.height(); i++) {  //Init frame and find max //opt min
		for (int e = 0; e < src.width(); e++) {
			*((*frame) + i * src.width() + e) = *(src.data() + i * src.width() + e);
		}
	}

	std::cout << filename << " loaded. Height: " << *height << " Width: " << *width << std::endl;
}