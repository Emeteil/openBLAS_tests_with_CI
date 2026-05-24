#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <cblas.h>

extern "C" { void openblas_set_num_threads(int num_threads); }

template <typename T>
void my_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const T alpha, const T *A, const int lda, const T *B, const int ldb, const T beta, T *C, const int ldc)
{
    if (Order == CblasRowMajor)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                T sum = 0.0;
                for (int p = 0; p < K; ++p)
                {
                    T a_val = (TransA == CblasTrans || TransA == CblasConjTrans) ? A[p*lda+i] : A[i*lda+p];
                    T b_val = (TransB == CblasTrans || TransB == CblasConjTrans) ? B[j*ldb+p] : B[p*ldb+j];
                    sum += a_val * b_val;
                }

                int idx = i*ldc+j;
                if (beta == 0.0)
                    C[idx] = alpha*sum;
                else
                    C[idx] = alpha*sum + beta*C[idx];
            }
        }
    }
    else
    {
        for (int j = 0; j < N; ++j)
        {
            for (int i = 0; i < M; ++i)
            {
                T sum = 0.0;
                for (int p = 0; p < K; ++p)
                {
                    T a_val = (TransA == CblasTrans || TransA == CblasConjTrans) ? A[i*lda+p] : A[p*lda+i];
                    T b_val = (TransB == CblasTrans || TransB == CblasConjTrans) ? B[p*ldb+j] : B[j*ldb+p];
                    sum += a_val * b_val;
                }

                int idx = j*ldc+i;
                if (beta == 0.0)
                    C[idx] = alpha*sum;
                else
                    C[idx] = alpha*sum + beta*C[idx];
            }
        }
    }
}

void openblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void openblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
void generate_random_matrix(std::vector<T>& mat, int size)
{
    std::mt19937 gen(54747543);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < size; ++i)
    {
        mat[i] = static_cast<T>(dist(gen));
    }
}

template <typename T>
void run_performance_test(int N)
{
    std::cout << "Выделение памяти для матриц размера " << N << "x" << N << "..." << std::endl;
    std::vector<T> A(N*N);
    std::vector<T> B(N*N);
    std::vector<T> C_my(N*N, 0.0);
    std::vector<T> C_openblas(N*N, 0.0);

    generate_random_matrix(A, N*N);
    generate_random_matrix(B, N*N);
    
    T alpha = 1.0;
    T beta = 0.0;
    
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    int runs = 4;
    
    std::cout << "Запуск my_gemm..." << std::endl;
    std::vector<double> my_times(runs);
    for (int r = 0; r < runs; ++r)
    {
        auto start_my = std::chrono::high_resolution_clock::now();
        my_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A.data(), N, B.data(), N, beta, C_my.data(), N);
        auto end_my = std::chrono::high_resolution_clock::now();
        my_times[r] = std::chrono::duration<double>(end_my - start_my).count();

        if (r == 0)
            std::cout << "  (Первый запуск my_gemm занял " << std::fixed << std::setprecision(4) << my_times[r] << " с)" << std::endl;
    }
    
    for (int threads : thread_counts)
    {
        std::cout << "\n=== Тестирование с количеством потоков: " << threads << " ===" << std::endl;
        openblas_set_num_threads(threads);
        
        std::vector<double> perf_ratios;
        
        for (int r = 0; r < runs; ++r)
        {
            double time_my = my_times[r];
            
            auto start_ob = std::chrono::high_resolution_clock::now();
            openblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A.data(), N, B.data(), N, beta, C_openblas.data(), N);
            auto end_ob = std::chrono::high_resolution_clock::now();
            double time_ob = std::chrono::duration<double>(end_ob - start_ob).count();
            
            double ratio = (time_ob / time_my) * 100.0;
            perf_ratios.push_back(ratio);
            
            std::cout << "  Прогон " << r + 1 << ": My_gemm = " << std::fixed << std::setprecision(4) << time_my << " с, "
                << "OpenBLAS = " << std::fixed << std::setprecision(4) << time_ob << " с, "
                << "Отн. произв. = " << std::fixed << std::setprecision(4) << ratio << " %"
            << std::endl;
        }
        
        double log_sum = 0.0;
        for (double r : perf_ratios)
        {
            log_sum += std::log(r);
        }
        double geom_mean = std::exp(log_sum / runs);
        
        std::cout << "Среднегеометрическая производительность (My_GEMM к OpenBLAS): " << std::fixed << std::setprecision(4) << geom_mean << " %" << std::endl;
    }
}

int main()
{
    int n_size = 2080;
    
    std::cout << "Размер матрицы: " << n_size << "x" << n_size << std::endl;
    
    std::cout << "\nFLOAT (Одинарная точность)" << std::endl;
    run_performance_test<float>(n_size);
    
    std::cout << "\nDOUBLE (Двойная точность)" << std::endl;
    run_performance_test<double>(n_size);

    return 0;
}