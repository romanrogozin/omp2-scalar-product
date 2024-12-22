#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <iomanip>


std::vector<double> prepare_vector(int& size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.1, 0.9);
    
    std::vector<double> vec(size);

    for (auto& val : vec) {
        val = dist(gen);
    }

    return vec;
}

double scalar_product(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double result = 0.0;
    int i;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

double scalar_product_chunked(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    int i;
    int chunk = 10;
    double result = 0.0;

    #pragma omp parallel for default(shared) private(i) schedule(static, chunk) reduction(+:result)
        for (i = 0; i < vec1.size(); i++)
            result += vec1[i] * vec2[i];

    return result;
}

int main() {
    std::vector<int> thread_experiments = { 1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024 };
    std::vector<int> vector_size_experiments = { 1'000'000, 10'000'000, 100'000'000, 500'000'000, 1'000'000'000 };
    int runs_count = 5;

    for (int i = 0; i < vector_size_experiments.size(); i++)
    {
        int vector_size_experiment = vector_size_experiments[i];
        std::vector<double> vec1 = prepare_vector(vector_size_experiment);
        std::vector<double> vec2 = prepare_vector(vector_size_experiment);

        for (int j = 0; j < thread_experiments.size(); j++)
        {
            int current_thread_experiment = thread_experiments[j];
            omp_set_num_threads(current_thread_experiment);
            //simple
            double total_execution_time = 0;
            double result = 0;
            for (int k = 0; k < runs_count; k++)
            {
                auto start = std::chrono::high_resolution_clock::now();

                result = scalar_product(vec1, vec2);

                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> duration = end - start;
                total_execution_time += duration.count();
            }
            double avg_exexution_time = total_execution_time / runs_count;

            //chunk based
            double total_execution_time_2 = 0;
            double result_2 = 0;
            for (int k = 0; k < runs_count; k++)
            {
                auto start = std::chrono::high_resolution_clock::now();

                result_2 = scalar_product_chunked(vec1, vec2);

                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> duration = end - start;
                total_execution_time_2 += duration.count();
            }
            double avg_exexution_time_2 = total_execution_time_2 / runs_count;
            
            double time_diff = avg_exexution_time - avg_exexution_time_2;

            std::cout << std::setprecision(10) << vector_size_experiment << ";" << current_thread_experiment << ";" << avg_exexution_time << ";" << avg_exexution_time_2 << ";" << time_diff << ";" << std::endl;
        }
    }

    std::cout << "Waiting for exit...";
    int temp;
    std::cin >> temp;
    return 0;
}