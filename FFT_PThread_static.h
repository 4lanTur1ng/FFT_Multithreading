#pragma once
#include<iostream>
#include<fstream>
#include <cmath>
#include <vector>
#include <pthread.h>
#include<semaphore.h>
#include <complex>
using namespace std;

// �߳�����
#define numThreads 4
#define PI 3.1415926535897932384626433832

struct ThreadParam0 {
    int start;
    int end;
    int step;
    int m;
    int t_id;
    vector<complex<double>>* input;
};

int flag = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

sem_t sem_main;                    
sem_t sem_workerstart[numThreads]; 
pthread_t threads[numThreads];
ThreadParam0 params[numThreads];

void* threadFunc0(void* arg)
{
    ThreadParam0* param = (ThreadParam0*)arg;
    vector<complex<double>>& input = *(param->input);

    while (1)
    {
        sem_wait(&sem_workerstart[param->t_id]);
        for (int i = param->start; i < param->end; i += param->step)//����ÿ���߳���˵�Ĳ���
        {
            complex<double> w(1);
            for (int j = 0; j < param->m; j++)
            {
                complex<double> t = w * input[i + j + param->m];//���ӻ��߱����Ĳ���
                input[i + j + param->m] = input[i + j] - t;//��벿��
                input[i + j] += t;//ǰ�벿��
                w *= complex<double>(cos(PI / param->m), -sin(PI / param->m));
            }
        }
        pthread_mutex_lock(&mutex);
        flag++;
        pthread_mutex_unlock(&mutex);   
        if (flag == numThreads) 
        {
            sem_post(&sem_main);
        }
        if (param->step >= input.size()/numThreads)
        {
            pthread_exit(NULL);
            return NULL;
        }
    }
}



void fft2_pthread_static(vector<complex<double>>& input)//pthread�汾
{
    int n = input.size();
    int step = n / numThreads;
    flag = 0;
    mutex = PTHREAD_MUTEX_INITIALIZER;
    sem_init(&sem_main, 0, 0);//�ź�����ʼ��
    for (int i = 0; i < numThreads; ++i)
    {
        sem_init(&sem_workerstart[i], 0, 0);
    }
    for (int i = 0; i < numThreads; i++)
    {
        params[i].start = i * step;
        params[i].end = (i == numThreads - 1) ? n : (i + 1) * step;
        params[i].input = &input;//����
        params[i].t_id = i;
        params[i].step = 0;
        params[i].m = 0;
        pthread_create(&threads[i], NULL, threadFunc0, &params[i]);
    }
    //��������
    for (int i = 1, j = 0; i < n; i++)
    {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }
    // ��������
    for (int k = 2; k <= n; k <<= 1)
    {
        int m = k >> 1;
        if (n / numThreads >= k)
        {
            for (int i = 0; i < numThreads; i++)
            {
                params[i].step = k;//��һ��k=2�������2
                params[i].m = m;//m=k/2
            }

            //���ѹ����߳�
            for (int i = 0; i < numThreads; i++)
            {
                sem_post(&sem_workerstart[i]);
            }
            sem_wait(&sem_main);
            flag = 0;
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

    for (int i = 0; i < numThreads; ++i)
    {
        sem_destroy(&sem_workerstart[i]);
    }
    for (int i = 0; i < numThreads; i++)
    {
        pthread_join(threads[i], NULL);
    }
}


