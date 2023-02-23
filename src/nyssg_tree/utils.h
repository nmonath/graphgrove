/*
 * Copyright (c) 2021 The authors of SG Tree All rights reserved.
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

#ifdef _FLOAT64_VER_
#define MY_NPY_FLOAT NPY_FLOAT64
#define PYTHON_FLOAT_CHAR "d"
typedef Eigen::MatrixXd matrixType;
typedef Eigen::VectorXd pointType;
typedef Eigen::VectorXd::Scalar scalar;
#else
#define MY_NPY_FLOAT NPY_FLOAT32
#define PYTHON_FLOAT_CHAR "f"
typedef Eigen::MatrixXf matrixType;
typedef Eigen::VectorXf pointType;
typedef Eigen::VectorXf::Scalar scalar;
#endif

#ifdef _MSC_VER

typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
typedef intptr_t ssize_t;

#endif

namespace utils
{

    inline void progressbar(size_t x, size_t n, unsigned int w = 50)
    {
        static std::mutex mtx;
        if ( (x != n) && (x % (n/10+1) != 0) ) return;

        float ratio =  x/(float)n;
        unsigned c = unsigned(ratio * w);

        if(mtx.try_lock())
        {
            std::cerr << std::setw(3) << (int)(ratio*100) << "% [";
            for (unsigned x=0; x<c; x++) std::cerr << "=";
            for (unsigned x=c; x<w; x++) std::cerr << " ";
            std::cerr << "]\r" << std::flush;
            mtx.unlock();
        }
    }

    template<class UnaryFunction>
    UnaryFunction parallel_for_progressbar(size_t first, size_t last, UnaryFunction f, unsigned cores=-1)
    {
        if (cores == -1) {
            cores = std::thread::hardware_concurrency();
        }
        // std::cerr << "Cores: " << cores << std::endl;
        const size_t total_length = last - first;
        const size_t chunk_length = total_length / cores;

        if(total_length <= 10000)
        {
            auto task = [&f,&chunk_length](size_t start, size_t end)->void{
                for (; start < end; ++start){
                    f(start);
                }
            };

            size_t chunk_start = first;
            std::vector<std::future<void>>  for_threads;
            for (unsigned i = 0; i < cores - 1; ++i)
            {
                const auto chunk_stop = chunk_start + chunk_length;
                for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
                chunk_start = chunk_stop;
            }
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

            for (auto& thread : for_threads)
                thread.get();
        }
        else
        {
            auto task = [&f,&chunk_length](size_t start, size_t end)->void{
                for (; start < end; ++start){
                    progressbar(start%chunk_length, chunk_length);
                    f(start);
                }
            };

            size_t chunk_start = first;
            std::vector<std::future<void>>  for_threads;
            for (unsigned i = 0; i < cores - 1; ++i)
            {
                const auto chunk_stop = chunk_start + chunk_length;
                for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
                chunk_start = chunk_stop;
            }
            for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

            for (auto& thread : for_threads)
                thread.get();
            progressbar(chunk_length, chunk_length);
            std::cerr << std::endl;
        }

        return f;
    }

    template<typename T>
    void add_to_atomic(std::atomic<T>& foo, T& bar)
    {
        auto current = foo.load(std::memory_order_relaxed);
        while (!foo.compare_exchange_weak(current, current + bar, std::memory_order_relaxed, std::memory_order_relaxed));
    }

    class ParallelAddMatrixNP
    {
        size_t left;
        size_t right;
        pointType res;
        const Eigen::Map<matrixType>& pMatrix;
        static std::atomic<int> objectCount;

        void run()
        {
            res = pointType::Zero(pMatrix.rows());
            for(size_t i = left; i<right; ++i)
                res += pMatrix.col(i);
        }

    public:
        ParallelAddMatrixNP(const Eigen::Map<matrixType>& pM) : pMatrix(pM)
        {
	    objectCount++;
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelAddMatrixNP(size_t left, size_t right, const Eigen::Map<matrixType>& pM) : pMatrix(pM)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelAddMatrixNP()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 500000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelAddMatrixNP* t1 = new ParallelAddMatrixNP(left, left + split, pMatrix);
            ParallelAddMatrixNP* t2 = new ParallelAddMatrixNP(left + split, right, pMatrix);

            std::future<int> f1 = std::async(std::launch::async, &ParallelAddMatrixNP::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelAddMatrixNP::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }

        pointType get_result()
        {
            return res;
        }
    };


    class ParallelDistanceComputeNP
    {
        size_t left;
        size_t right;
        pointType res;
        pointType& vec;
        const Eigen::Map<matrixType>& pMatrix;
        static std::atomic<int> objectCount;

        void run()
        {
            res = pointType::Zero(pMatrix.cols());
            for(size_t i = left; i<right; ++i)
                res[i] = std::exp(-pMatrix.col(i).dot(vec));
        }

    public:
        ParallelDistanceComputeNP(const Eigen::Map<matrixType>& pM, pointType& v) : vec(v), pMatrix(pM)
        {
	    objectCount++;
            this->left = 0;
            this->right = pM.cols();
            compute();
        }
        ParallelDistanceComputeNP(size_t left, size_t right, const Eigen::Map<matrixType>& pM, pointType& v) : vec(v), pMatrix(pM)
        {
	    objectCount++;
            this->left = left;
            this->right = right;
        }

        ~ParallelDistanceComputeNP()
        {
	    objectCount--;
	}

        int compute()
        {
            if ((right - left < 10000) || (objectCount.load() > 128))
            {
                run();
                return 0;
            }

            size_t split = (right - left) / 2;

            ParallelDistanceComputeNP* t1 = new ParallelDistanceComputeNP(left, left + split, pMatrix, vec);
            ParallelDistanceComputeNP* t2 = new ParallelDistanceComputeNP(left + split, right, pMatrix, vec);

            std::future<int> f1 = std::async(std::launch::async, &ParallelDistanceComputeNP::compute, t1);
            std::future<int> f2 = std::async(std::launch::async, &ParallelDistanceComputeNP::compute, t2);

            f1.get();
            f2.get();

            res = t1->res + t2->res;

            delete t1;
            delete t2;

            return 0;
        }

        pointType get_result()
        {
            return res;
        }
    };

    struct ParsedArgs
    {
        unsigned K;
        unsigned n_iters;
        unsigned n_save;
        unsigned n_threads;
        unsigned n_top_words;
        std::string algo;
        std::string init_type;
        std::string data_path;
        std::string name;
        std::string out_path;

        ParsedArgs(int argc, char ** argv)
        {
            // set default values
            K = 100;
            n_iters = 1000;
            n_save = 200;
            n_threads = std::thread::hardware_concurrency();
            n_top_words = 15;
            algo = "simple";
            data_path = "./";
            name = "";
            out_path = "./";

            // iterate
            std::vector<std::string> arguments(argv, argv + argc);
            for (auto arg = arguments.begin(); arg != arguments.end(); ++arg)
            {
                if (*arg == "--method")
                {
                    algo = *(++arg);
                    if (algo == "")
                        algo = "simple";
                }
                else if (*arg == "--output-model")
                {
                    out_path = *(++arg);
                    if (out_path == "")
                        out_path = "./";
                }
                else if (*arg == "--init-type")
                {
                    init_type = *(++arg);
                    if (init_type == "")
                        init_type = "random";
                }
                else if (*arg == "--dataset")
                {
                    name = *(++arg);
                    if (name == "")
                        std::cout << "Error: Invalid file path to training corpus.";
                    std::string::size_type idx = name.find_last_of("/");
                    if (idx == std::string::npos)
                    {
                        data_path = "./";
                    }
                    else
                    {
                        data_path = name.substr(0, idx + 1);
                        name = name.substr(idx + 1, name.size() - data_path.size());
                    }
                }
                else if (*arg == "--num-clusters")
                {
                    int _K = std::stoi(*(++arg));
                    if (_K > 0)
                        K = _K;
                    else
                        std::cout << "Error: Invalid number of clusters.";
                }
                else if (*arg == "--num-topics")
                {
                    int _K = std::stoi(*(++arg));
                    if (_K > 0)
                        K = _K;
                    else
                        std::cout << "Error: Invalid number of topics.";
                }
                else if (*arg == "--num-iterations")
                {
                    int _n_iters = std::stoi(*(++arg));
                    if (_n_iters > 0)
                        n_iters = _n_iters;
                    else
                        std::cout << "Error: Invalid number of iterations.";
                }
                else if (*arg == "--num-top-words")
                {
                    int _n_top_words = std::stoi(*(++arg));
                    if (_n_top_words > 0)
                        n_top_words = _n_top_words;
                    else
                        std::cout << "Error: Invalid number of top words.";
                }
                else if (*arg == "--num-threads")
                {
                    int _n_threads = std::stoi(*(++arg));
                    if(_n_threads > 0)
                        n_threads = _n_threads;
                    else
                        std::cout << "Error: Invalid number of threads.";
                }
                else if (*arg == "--output-state-interval")
                {
                    int _n_save = std::stoi(*(++arg));
                    if (_n_save > 0)
                        n_save = _n_save;
                    else
                        std::cout << "Error: Invalid output state interval.";
                }
            }
        }

        ParsedArgs(unsigned num_atoms=100, unsigned num_iters=1000, std::string algorithm="simple",
                   unsigned output_interval=200, unsigned top_words=15,
                   unsigned num_threads=std::thread::hardware_concurrency(),
                   std::string init_scheme="random", std::string output_path="./")
        {
            K = num_atoms;
            n_iters = num_iters;
            n_save = output_interval;
            n_threads = num_threads;
            n_top_words = top_words;
            algo = algorithm;
            init_type = init_scheme;
            data_path = "./";
            name = "custom";
            out_path = output_path;
        }
    };
}


#endif 
