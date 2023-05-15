#pragma once
#include <cmath>
#include <complex>
#include <vector>
#include <pthread.h>
#include<iostream>
#include<fstream>

// 线程数量
#define numThreads 4
#define PI 3.1415926535897932384626433832

using namespace std;

// 定义任务结构体
struct ThreadParam {
    vector<complex<double>>* input;
    int start;
    int end;
    int step;
    int m;
    complex<double> w_m;
};

// 定义线程函数
void* threadFunc(void* arg)
{
    ThreadParam* param = (ThreadParam*)arg;
    vector<complex<double>>& input = *(param->input);
    for (int i = param->start; i < param->end; i += param->step)
    {
        complex<double> w(1);
        for (int j = 0; j < param->m; j++)
        {
            complex<double> t = w * input[i + j + param->m];
            input[i + j + param->m] = input[i + j] - t;
            input[i + j] += t;
            w *= complex<double>(cos(PI / param->m), -sin(PI / param->m));
        }
    }
    return NULL;
}

void fft2_pthread(vector<complex<double>>& input)
{
    int n = input.size();
    for (int i = 1, j = 0; i < n; i++)
    {   // 数据重排
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }

    for (int k = 2; k <= n; k <<= 1)
    {   // 蝴蝶运算
        int m = k >> 1;
        pthread_t threads[numThreads];
        ThreadParam params[numThreads];
        int step = n / numThreads;
        if (n / numThreads > k)
        {
            for (int i = 0; i < numThreads; i++)
            {
                params[i].start = i * step;
                params[i].end = (i == numThreads - 1) ? n : (i + 1) * step;
                params[i].step = k;
                params[i].m = m;
                params[i].input = &input;
                pthread_create(&threads[i], NULL, threadFunc, (void*)&params[i]);
            }
            for (int i = 0; i < numThreads; i++)pthread_join(threads[i], NULL);
        }
        else
        {
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
}

