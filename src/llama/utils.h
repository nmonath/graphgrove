/* Modified from https://github.com/manzilzaheer/CoverTree
 * Copyright (c) 2017 Manzil Zaheer All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _UTILS_H
#define _UTILS_H

#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <atomic>
#include <thread>
#include <future>

#include <Eigen/Core>

#ifdef _MSC_VER

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;

#endif

namespace utils
{

    // timing methods
    static std::chrono::time_point<std::chrono::high_resolution_clock> get_time()
    {
        return std::chrono::high_resolution_clock::now();
    }

    static float timedur(std::chrono::time_point<std::chrono::high_resolution_clock> st,
               std::chrono::time_point<std::chrono::high_resolution_clock> en)
    {
        return (float) std::chrono::duration_cast<std::chrono::microseconds>(en - st).count() / (float) 1000000.0;
    }

    static long timedur_long(std::chrono::time_point<std::chrono::high_resolution_clock> st,
               std::chrono::time_point<std::chrono::high_resolution_clock> en)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(en - st).count(); // / (float) 1000000.0;
    }


    static inline void pause()
    {
        // Only use this function if a human is involved!
        std::cout << "Press any key to continue..." << std::flush;
        std::cin.get();
    }

    template<class InputIt, class UnaryFunction>
    UnaryFunction parallel_for_each(InputIt first, InputIt last, UnaryFunction f)
    {
        unsigned cores = std::thread::hardware_concurrency();

        auto task = [&f](InputIt start, InputIt end)->void{
            for (; start < end; ++start)
                f(*start);
        };

        const size_t total_length = std::distance(first, last);
        const size_t chunk_length = total_length / cores;
        InputIt chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < cores - 1; ++i)
        {
            const auto chunk_stop = std::next(chunk_start, chunk_length);
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }

    template<class InputIt, class UnaryFunction>
    UnaryFunction parallel_for_each(unsigned cores, InputIt first, InputIt last, UnaryFunction f)
    {

        auto task = [&f](InputIt start, InputIt end)->void{
            for (; start < end; ++start)
                f(*start);
        };

        const size_t total_length = std::distance(first, last);
        const size_t chunk_length = total_length / cores;
        InputIt chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < cores - 1; ++i)
        {
            const auto chunk_stop = std::next(chunk_start, chunk_length);
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }


    template<class UnaryFunction>
    UnaryFunction parallel_for(size_t first, size_t last, UnaryFunction f)
    {
        if (first >= last) {
            return f;
        }

        unsigned cores = std::thread::hardware_concurrency();

        auto task = [&f](size_t start, size_t end)->void{
            for (; start < end; ++start)
                f(start);
        };

        const size_t total_length = last - first;
        const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));
        size_t chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
        {
            const auto chunk_stop = chunk_start + chunk_length;
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }

    template<class UnaryFunction>
    UnaryFunction parallel_for(unsigned cores, size_t first, size_t last, UnaryFunction f)
    {
        if (first >= last) {
            return f;
        }

        auto task = [&f](size_t start, size_t end)->void{
            for (; start < end; ++start)
                f(start);
        };

        const size_t total_length = last - first;
        const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));
        size_t chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
        {
            const auto chunk_stop = chunk_start + chunk_length;
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        return f;
    }


    static inline void progressbar(unsigned int x, unsigned int n, unsigned int w = 50){
        if ( (x != n) && (x % (n/10+1) != 0) ) return;

        float ratio =  x/(float)n;
        unsigned c = ratio * w;

        std::cout << std::setw(3) << (int)(ratio*100) << "% [";
        for (unsigned x=0; x<c; x++) std::cout << "=";
        for (unsigned x=c; x<w; x++) std::cout << " ";
        std::cout << "]\r" << std::flush;
    }

    template<class UnaryFunction>
    UnaryFunction parallel_for_progressbar(size_t first, size_t last, UnaryFunction f)
    {
        if (first >= last) {
            return f;
        }

        unsigned cores = std::thread::hardware_concurrency();
        const size_t total_length = last - first;
        const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));

        auto task = [&f,&chunk_length](size_t start, size_t end)->void{
            for (; start < end; ++start){
                progressbar(start%chunk_length, chunk_length);
                f(start);
            }
        };

        size_t chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
        {
            const auto chunk_stop = chunk_start + chunk_length;
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        progressbar(chunk_length, chunk_length);
        std::cout << std::endl;
        return f;
    }

    template<class UnaryFunction>
    UnaryFunction parallel_for_progressbar(unsigned cores, ssize_t first, size_t last, UnaryFunction f)
    {
        if (first >= last) {
            return f;
        }

//        unsigned cores = std::thread::hardware_concurrency();
        const size_t total_length = last - first;
        const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));

        auto task = [&f,&chunk_length](size_t start, size_t end)->void{
            for (; start < end; ++start){
                progressbar(start%chunk_length, chunk_length);
                f(start);
            }
        };

        size_t chunk_start = first;
        std::vector<std::future<void>>  for_threads;
        for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
        {
            const auto chunk_stop = chunk_start + chunk_length;
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
            chunk_start = chunk_stop;
        }
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

        for (auto& thread : for_threads)
            thread.get();
        progressbar(chunk_length, chunk_length);
        std::cout << std::endl;
        return f;
    }

    template<typename T>
    void add_to_atomic(std::atomic<T>& foo, T& bar)
    {
        auto current = foo.load();
        while (!foo.compare_exchange_weak(current, current + bar));
    }

}


#endif
