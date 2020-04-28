#pragma once
#include <string>
#include <chrono>

class Timer
{
public:
	Timer(std::string timer_for);
	~Timer();

private:
	std::chrono::time_point<std::chrono::system_clock> m_start;
	std::string m_timer_for;
};

void save_labels(float *labels, int width, int height, std::string filename);
void load_image(std::string filename, float **frame, int *width, int *height);