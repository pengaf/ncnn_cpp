#include <chrono>
#include <array>
#include <iostream>
#include <ncnn/net.h>
#include <ncnn/mat.h>

void test()
{
	ncnn::Net net;
	net.load_param("g:/model/test2.param");
	net.load_model("g:/model/test2.bin");


	std::array<float, 3 * 11 * 11> input_data{};

	ncnn::Mat in(11, 11, 3);// input blob as above
	in.fill(1.0f);
	std::vector<float> scores;

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	ncnn::Mat out, out2;
	for (uint32_t i = 0; i < 14000; ++i)
	{
		ncnn::Extractor ex = net.create_extractor();
		int a = ex.input("input", in);
		int b = ex.extract("value", out);
		//ex.extract("policy", out2);
		ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
		for (int j = 0; j < out_flatterned.w; j++)
		{
			scores.push_back(out_flatterned[j]);
		}
		//in.fill(i*1.0f);

		//for (int c = 0; c < 3; ++c)
		//{
		//	ncnn::Mat ch = in.channel(c);
		//	for (int y = 0; y < 11; ++y)
		//	{
		//		float* d = ch.row(y);
		//		for (int x = 0; x < 11; ++x)
		//		{
		//			d[x] = 1.0f;
		//		}
		//	}
		//}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::duration startDuration = end - start;
	double seconds = startDuration.count() * 0.000000001;
	printf("%f", seconds);
}

void testvk()
{
	//ncnn::create_gpu_instance();

	ncnn::Net net;
	//net.opt.num_threads = 1;
	//net.opt.use_vulkan_compute = 1;
	//net.opt.use_winograd_convolution = 1;
	net.load_param("g:/model/test2.param");
	net.load_model("g:/model/test2.bin");


	//std::array<float, 3 * 11 * 11> input_data{};

	ncnn::Mat in(11, 11, 2, 3);// input blob as above
	in.fill(1.0f);
	std::vector<float> scores;

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	ncnn::Mat out, out2;
	for (uint32_t i = 0; i < 14000; ++i)
	{
		ncnn::Extractor ex = net.create_extractor();
		//ex.set_vulkan_compute(true);
		int a = ex.input("input", in);
		int b = ex.extract("value", out);
		//ex.extract("policy", out2);
		//ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
		//for (int j = 0; j < out_flatterned.w; j++)
		//{
		//	scores.push_back(out_flatterned[j]);
		//}
		//in.fill(i*1.0f);

		//for (int c = 0; c < 3; ++c)
		//{
		//	ncnn::Mat ch = in.channel(c);
		//	for (int y = 0; y < 11; ++y)
		//	{
		//		float* d = ch.row(y);
		//		for (int x = 0; x < 11; ++x)
		//		{
		//			d[x] = 1.0f;
		//		}
		//	}
		//}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::duration startDuration = end - start;
	double seconds = startDuration.count() * 0.000000001;
	printf("%f", seconds);
}

int main() 
{
	try
	{
		testvk();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}

}