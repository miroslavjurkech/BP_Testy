
#define ONLY_IMPORTANT
#include"my_functions.h"

#include "./CUCCL_LE/CUCCL_LE.cuh"
#include "./CUCCL_NP/CUCCL_NP.cuh"
#include "./CUCCL_DPL/CUCCL_DPL.cuh"



#include <iomanip>
#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace std; 
using namespace CUCCL; 


void testCCL(char const* flag, int *data, const int width, const int height, int degreeOfConnectivity, int threshold, int *labels)
{
#ifndef ONLY_IMPORTANT
   // const auto width = 32;
// 	const auto height = 8;
// 	unsigned char data[width * height] =
// 	{
// 		135, 135, 240, 240, 240, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 135, 120, 120,
// 		135, 135, 240, 240, 240, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 135, 120, 120,
// 		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 135, 120, 120,
// 		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 135, 135, 135, 120, 120, 120,
// 		135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
// 		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
// 		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
// 		135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120
//     };


    cout << "Binary image is : " <<endl;
	for (auto i = 0; i < height; i++)
	{
		for (auto j = 0; j < width; j++)
		{
			cout << setw(3) << static_cast<int>(data[i * width + j]) << " ";
		}
		cout << endl;
	}
    cout<<endl;
#endif // !ONLY_IMPORTANT
    
	if (flag == (std::string)"LE")
	{
		CCLLEGPU ccl;

		ccl.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold);

#ifndef ONLY_IMPORTANT

		cout << "Label Mesh by CCL LE : " << endl;
		for (auto i = 0; i < height; i++)
		{
			for (auto j = 0; j < width; j++)
			{
				dest[i * width + j] = labels[i * width + j];
				//cout << setw(3) << labels[i * width + j] << " ";

			}
			//cout << endl;
		}
#endif // !ONLY_IMPORTANT
	}
    

    if (flag == (std::string)"NP")
    {
        CCLNPGPU cclnp;
	    cclnp.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold);

#ifndef ONLY_IMPORTANT

	    cout << "Label Mesh by CCL NP : " << endl;
	    for (auto i = 0; i < height; i++)
	    {
		    for (auto j = 0; j < width; j++)
		    {
			    cout << setw(3) << labels[i * width + j] << " ";
		    }
		    cout << endl;
	    }
#endif // !ONLY_IMPORTANT
	}

	if (flag == (std::string)"DPL")
	{
		CCLDPLGPU ccldpl;
		ccldpl.CudaCCL(data, labels, width, height, degreeOfConnectivity, threshold);

#ifndef ONLY_IMPORTANT
		cout << "Label Mesh by CCL DPL : " << endl;
		for (auto i = 0; i < height; i++)
		{
			for (auto j = 0; j < width; j++)
			{
				cout << setw(3) << labels[i * width + j] << " ";
			}
			cout << endl;
		}
#endif // !ONLY_IMPORTANT
	}
}


int main(int argc, char **args)
{
	if (argc != 6) {
		std::cerr << "Number of arguments is invalid" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (args[4][0] != '4' && args[4][0] != '8') {
		std::cerr << "Connectivity (4. argument) not set correctly" << std::endl;
		exit(EXIT_FAILURE);
	}
	int connectivity = args[4][0] - '0';
	
	int threshold = atoi(args[5]);
	if (args[5][0] < '0' || args[5][0] > '9' || threshold < 0) {
		std::cerr << "Threshold (5. argument) not set correctly" << std::endl;
		exit(EXIT_FAILURE);
	}

	int *img_map;
	int width, height;
	load_image(args[1], &img_map, &width, &height);

	int *labels = new int[width * height];
	if (labels == NULL) {
		std::cerr << "Label array allocation failed" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaFree(0);

	{
		Timer my_time(std::string(args[3]) + ": In total ");
		testCCL(args[3], img_map, width, height, connectivity, threshold, labels);
	}
	std::cout << "Number of final segments " << count_segments(labels, width, height) << ".\n";

	save_labels(labels, width, height, args[2]);

	std::cout << "DONE" << std::endl;

	delete[] img_map;
	delete[] labels;
}
