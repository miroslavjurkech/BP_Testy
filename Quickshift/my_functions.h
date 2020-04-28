#pragma once
#include <string>
#include <chrono>
#include "quickshift_common.h"

class Timer
{
public:
	Timer(std::string timer_for);
	~Timer();

private:
	std::chrono::time_point<std::chrono::system_clock> m_start;
	std::string m_timer_for;
};

int count_segments(int *frame, int width, int height);
void save_labels(int *labels, int width, int height, std::string filename);
void load_image(std::string filename, float **frame, int *width, int *height);
void cimg_to_matlab(const std::string& name, image_t& im);