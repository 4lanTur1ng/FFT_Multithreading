#include <iostream>
#include <vector>
#include <immintrin.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <time.h>
#include <omp.h>
#include <windows.h>
#include <pthread.h>
#include"FFT_PThread.h"
#include"FFT_PThread_static.h"

using namespace std;

#define PI 3.1415926535897932384626433832
bool parallel = true;
#define NUM_THREADS 4

struct COMPLEX
{
	double real = 0;
	double imag = 0;
};

template<typename T>
void separate(T* a, size_t n)
{
	if (n < 2)
	{
		return;
	}
	else
	{
		T* b = new T[n / 2];				// get temp heap storage
		for (int i = 0; i < n / 2; i++)		// copy all odd elements to heap storage
			b[i] = a[i * 2 + 1];
		for (int i = 0; i < n / 2; i++)		// copy all even elements to lower-half of a[]
			a[i] = a[i * 2];
		for (int i = 0; i < n / 2; i++)		// copy all odd (from heap) to upper-half of a[]
			a[i + n / 2] = b[i];
		delete[] b;							// delete heap storage
	}
}

void fft(vector<double> input, vector<complex<double>>& output)
{
	// 串行fft算法
	size_t length = input.size();
	if (length >= 2)
	{
		// 分为奇偶
		vector<double> odd;
		vector<double> even;
		for (size_t n = 0; n < length; n++)
		{
			if (n & 1)
			{
				odd.push_back(input.at(n));
			}
			else
			{
				even.push_back(input.at(n));
			}
		}
		// 重排
		// 低
		vector<complex<double>> fft_even_out(output.begin(), output.begin() + length / 2);
		// 高
		vector<complex<double>> fft_odd_out(output.begin() + length / 2, output.end());
		// 递归执行代码
		fft(even, fft_even_out);
		fft(odd, fft_odd_out);

		// 组合奇偶部分
		complex<double> odd_out;
		complex<double> even_out;
		for (size_t n = 0; n != length / 2; n++)
		{
			if (length == 2)
			{
				even_out = even[n] + fft_even_out[n];
				odd_out = odd[n] + fft_odd_out[n];
			}
			else
			{
				even_out = fft_even_out[n];
				odd_out = fft_odd_out[n];
			}
			// 翻转因子
			complex<double> w = exp(complex<double>(0, -2.0 * PI * double(n) / (double)(length)));
			// even part
			output[n] = even_out + w * odd_out;
			// odd part
			output[n + length / 2] = even_out - w * odd_out;
		}
	}
}

void fft2(vector<complex<double>>& input) 
{
	int n = input.size();

	// 数据重排
	for (int i = 1, j = 0; i < n; i++) 
	{
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// 蝴蝶运算
	for (int k = 2; k <= n; k <<= 1) 
	{
		int m = k >> 1;
		complex<double> w_m(cos(PI / m), -sin(PI / m));

		for (int i = 0; i < n; i += k) 
		{
			complex<double> w(1);
			for (int j = 0; j < m; j++) 
			{
				complex<double> t = w * input[i + j + m];
				input[i + j + m] = input[i + j] - t;
				input[i + j] += t;
				w *= w_m;
			}
		}
	}
}

void fft2_omp(vector<complex<double>>& input)
{
	int n = input.size();

	// 数据重排
	for (int i = 1, j = 0; i < n; i++)
	{
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// 蝴蝶运算
	for (int k = 2; k <= n; k <<= 1)
	{
		int m = k >> 1;
		complex<double> w_m(cos(PI / m), -sin(PI / m));
#pragma omp parallel for num_threads(4)
		for (int i = 0; i < n; i += k)
		{
			complex<double> w(1);
			for (int j = 0; j < m; j++)
			{
				complex<double> t = w * input[i + j + m];
				input[i + j + m] = input[i + j] - t;
				input[i + j] += t;
				w *= w_m;
			}
		}
	}
}

void fft2_omp2(vector<complex<double>>& input)
{
	int n = input.size();

	// 数据重排
	for (int i = 1, j = 0; i < n; i++)
	{
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// 蝴蝶运算
complex<double> w(1);	
#pragma omp parallel num_threads(4) private(w)
	{

		for (int k = 2; k <= n; k <<= 1)
		{
			int m = k >> 1;
			complex<double> w_m(cos(PI / m), -sin(PI / m));

#pragma omp for 
			for (int i = 0; i < n; i += k)
			{
				for (int j = 0; j < m; j++)
				{
					complex<double> t = w * input[i + j + m];
					input[i + j + m] = input[i + j] - t;
					input[i + j] += t;
					w *= w_m;
				}
			}
		}
	}
}

void fft2_omp2_simd(COMPLEX *input, int length)
{
	// 数据重排
	for (int i = 1, j = 0; i < length; i++)
	{
		int bit = length >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// 蝴蝶运算
#pragma omp parallel num_threads(4)
	{

		for (int k = 2; k <= length; k <<= 1)
		{
			int m = k >> 1;
#pragma omp for 
			for (int i = 0; i < length; i += k)
			{
				for (int j = 0; j < m; j++)
				{
					// complex<double> t = w * input[i + j + m];
					// compute t
					__m128d wr = _mm_set1_pd(cos(j*PI / m));
					__m128d wi = _mm_set_pd(-sin(j*PI / m), sin(j*PI / m));
					__m128d o = _mm_load_pd((double*)&input[i + j + m]);	// odd a|b
					__m128d e = _mm_load_pd((double*)&input[i + j]);	// even
					wr = _mm_mul_pd(o, wr);	// a*c|b*c
					__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
					wi = _mm_mul_pd(n1, wi);
					n1 = _mm_add_pd(wr, wi);
					wr = _mm_add_pd(e, n1);
					wi = _mm_sub_pd(e, n1);
					// input[i + j + m] = input[i + j] - t;
					_mm_store_pd((double*)&input[i + j + m],wi);
					// input[i + j] += t;
					_mm_store_pd((double*)&input[i + j], wr);
				}
			}
		}
	}
}

void fft2_simd(COMPLEX* input, int length)
{
	// 数据重排
	for (int i = 1, j = 0; i < length; i++)
	{
		int bit = length >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// 蝴蝶运算
	{

		for (int k = 2; k <= length; k <<= 1)
		{
			int m = k >> 1;
			for (int i = 0; i < length; i += k)
			{
				for (int j = 0; j < m; j++)
				{
					// complex<double> t = w * input[i + j + m];
					// compute t
					__m128d wr = _mm_set1_pd(cos(j * PI / m));
					__m128d wi = _mm_set_pd(-sin(j * PI / m), sin(j * PI / m));
					__m128d o = _mm_load_pd((double*)&input[i + j + m]);	// odd a|b
					__m128d e = _mm_load_pd((double*)&input[i + j]);	// even
					wr = _mm_mul_pd(o, wr);	// a*c|b*c
					__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
					wi = _mm_mul_pd(n1, wi);
					n1 = _mm_add_pd(wr, wi);
					wr = _mm_add_pd(e, n1);
					wi = _mm_sub_pd(e, n1);
					// input[i + j + m] = input[i + j] - t;
					_mm_store_pd((double*)&input[i + j + m], wi);
					// input[i + j] += t;
					_mm_store_pd((double*)&input[i + j], wr);
				}
			}
		}
	}
}

void fft_sse2(COMPLEX* input, size_t length)
{
	if (length >= 2)
	{
		separate(input, length);
		fft_sse2(input, length / 2);
		fft_sse2(input + length / 2, length / 2);

		for (size_t i = 0; i < length / 2; i++) {

			__m128d o = _mm_load_pd((double*)&input[i + length / 2]);   // odd a|b
			double SIN = sin(-2. * PI * i / length);
			__m128d wr = _mm_set1_pd(cos(-2. * PI * i / length));			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
			__m128d wi = _mm_set_pd(SIN, -SIN);		// -d | d	, note that it is reverse order
			// compute the w*o
			wr = _mm_mul_pd(o, wr);					// ac|bc
			__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1)); // invert  bc|ac
			wi = _mm_mul_pd(n1, wi);				// -bd|ad
			n1 = _mm_add_pd(wr, wi);				// ac-bd|bc+ad 

			o = _mm_load_pd((double*)&input[i]);	// load even part
			wr = _mm_add_pd(o, n1);					// compute even part, X_e + w * X_o;
			wi = _mm_sub_pd(o, n1);					// compute odd part,  X_e - w * X_o;
			_mm_store_pd((double*)&input[i], wr);
			_mm_store_pd((double*)&input[i + length / 2], wi);
		}
	}
	else
	{
		// do nothing
	}
}

void fft_avx512(COMPLEX* input, size_t length)
{
	if (length >= 8)
	{
		// 重排
		separate(input, length);
		// 递归执行代码
		fft_avx512(input, length / 2);
		fft_avx512(input + length / 2, length / 2);
		for (int i = 0; i < length / 2; i += 4)
		{
			// 优化
			__m512d o = _mm512_load_pd((double*)&input[i + length / 2]);   // odd a|b
			__m512d angle = _mm512_set_pd(
				-2. * PI * (i + 3) / length, 2. * PI * (i + 3) / length,
				-2. * PI * (i + 2) / length, 2. * PI * (i + 2) / length,
				-2. * PI * (i + 1) / length, 2. * PI * (i + 1) / length,
				-2. * PI * i / length, 2. * PI * i / length
			);

			__m512d wr = _mm512_cos_pd(angle);			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
			__m512d wi = _mm512_sin_pd(angle);		// -d | d	, note that it is reverse order
			// compute the w*o
			wr = _mm512_mul_pd(o, wr);					// ac|bc
			__m512d n1 = _mm512_shuffle_pd(o, o, 0x55); // invert  bc|ac
			wi = _mm512_mul_pd(n1, wi);				// -bd|ad
			n1 = _mm512_add_pd(wr, wi);				// ac-bd|bc+ad 

			o = _mm512_load_pd((double*)&input[i]);	// load even part
			wr = _mm512_add_pd(o, n1);					// compute even part, X_e + w * X_o;
			wi = _mm512_sub_pd(o, n1);					// compute odd part,  X_e - w * X_o;
			_mm512_store_pd((double*)&input[i], wr);
			_mm512_store_pd((double*)&input[i + length / 2], wi);
		}
	}
	else
	{
		// do nothing
		if (length >= 2)
		{
			separate(input, length);
			fft_sse2(input, length / 2);
			fft_sse2(input + length / 2, length / 2);
			for (int i = 0; i < length / 2; i++)
			{
				__m128d o = _mm_load_pd((double*)&input[i + length / 2]);   // odd
				__m128d wr = _mm_set1_pd(cos(-2. * PI * i / length));			//__m128d wr =  _mm_set_pd( cc,cc );		// cc 
				__m128d wi = _mm_set_pd(sin(-2. * PI * i / length), -sin(-2. * PI * i / length));		// -d | d	, note that it is reverse order
				// compute the w*o
				wr = _mm_mul_pd(o, wr);					// ac|bc
				__m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1)); // invert
				wi = _mm_mul_pd(n1, wi);				// -bd|ad
				n1 = _mm_add_pd(wr, wi);				// ac-bd|bc+ad 

				o = _mm_load_pd((double*)&input[i]);	// load even part
				wr = _mm_add_pd(o, n1);					// compute even part, x_e + w * x_o;
				wi = _mm_sub_pd(o, n1);					// compute odd part,  x_e - w * x_o;
				_mm_store_pd((double*)&input[i], wr);
				_mm_store_pd((double*)&input[i + length / 2], wi);
			}
		}
		else
		{
			return;
		}
	}
}


COMPLEX* input = new COMPLEX[1048576];
//vector<complex<double>> result(65536);	// 结果

void test(vector<double>& data)
{
	// 测试函数，每个算法循环测试counter
	// 循环测试次数
	size_t counter = 1;
	size_t size = data.size();
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	double quadpart = (double)frequency.QuadPart;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	//QueryPerformanceCounter(&start);
	//for (size_t n = 0; n < counter; n++) 
	//{
	//	fft(data,result);
	//}
	//QueryPerformanceCounter(&end);
	//cout << "fft cost : " << (end.QuadPart - start.QuadPart) / quadpart * 1000 / counter << " ms.\n";

	//// fft2
	//// load data
	vector<complex<double>> input2(size);
	for (size_t i = 0; i < size; i++)
	{
		input[i].real = data[i];
		input[i].imag = 0;
		input2[i] = complex<double>(data[i],0);
	}
	//QueryPerformanceCounter(&start);
	//for (size_t n = 0; n < counter; n++)
	//{
	//	fft_sse2(input, size);
	//}
	//QueryPerformanceCounter(&end);
	//cout << "fft_sse2 cost : " << (end.QuadPart - start.QuadPart) / quadpart / counter * 1000 << " ms.\n";

	//QueryPerformanceCounter(&start);
	//for (size_t n = 0; n < counter; n++)
	//{
	//	fft2_omp2(input2);
	//}
	//QueryPerformanceCounter(&end);
	//cout << "fft2 cost : " << (end.QuadPart - start.QuadPart) / quadpart / counter * 1000 << " ms.\n";

	//QueryPerformanceCounter(&start);
	//for (size_t n = 0; n < counter; n++)
	//{
	//	fft2_omp2_simd(input, size);
	//}
	//QueryPerformanceCounter(&end);
	//cout << "fft2_simd cost : " << (end.QuadPart - start.QuadPart) / quadpart / counter * 1000 << " ms.\n";
	QueryPerformanceCounter(&start);
	for (size_t n = 0; n < counter; n++)
	{
		fft2_pthread(input2);
	}
	QueryPerformanceCounter(&end);
	cout << "fft2_pthread cost : " << (end.QuadPart - start.QuadPart) / quadpart / counter * 1000 << " ms.\n";
}


int main() {
	// 读取数据集
	// 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
	ifstream fi("fft_1048576.txt");
	vector<double> data;
	string read_temp;
	while (fi.good())
	{
		getline(fi, read_temp);
		data.push_back(stod(read_temp));
	}
	test(data);
	//vector<complex<double>> input0(data.size());
	////// 验证结果的正确性
	//ofstream fo;
	//for (size_t i = 0; i < data.size(); i++)
	//{
	//	input[i].real = data[i];
	//	input[i].imag = 0;
	//	input3[i] = complex<double>(data[i], 0);
	//	input0[i] = complex<double>(data[i], 0);
	//}
	//fft2_pthread(input0);
	//fo.open("fft2_pthread_static_result.txt", ios::out);
	//for (int i = 0; i < input0.size(); i++)
	//{
	//	fo << input0[i] << endl;
	//}

	//fft(data, result);
	//fo.open("fft_result.txt", ios::out);
	//for (int i = 0; i < result.size(); i++)
	//{
	//	fo << result[i] << endl;
	//}

	//COMPLEX *input = new COMPLEX[1024];
	//for (size_t i = 0; i < 1024; i++)
	//{
	//	input[i].real = data[i];
	//	input[i].imag = 0;
	//}
	//fft2_omp2_simd(input, 1024);
	//fo.open("fft_omp_simd_result.txt", ios::out);
	//for (int i = 0; i < 1024; i++)
	//{
	//	fo <<'(' << input[i].real<<','<< input[i].imag << ')' << endl;
	//}
	//fo.close();
	fi.close();
	return 0;
}
