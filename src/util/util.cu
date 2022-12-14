
#include "util/util.cuh"

#include <iostream>

extern size_t db_bytes;
extern size_t db_max_bytes;
extern size_t db_layer_max_bytes;

void log_print(std::string str) {
#if (LOG_DEBUG)
    std::cout << "----------------------------" << std::endl;
    std::cout << "Started " << str << std::endl;
    std::cout << "----------------------------" << std::endl;	
#endif
}

void error(std::string str) {
    std::cout << "Error: " << str << std::endl;
	exit(-1);
}

void printMemUsage() {

    size_t free_byte;
    size_t total_byte;
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    LOG_F(INFO, "Memory usage on device %i: used = %f, free = %f, total = %f MiB",
            device_id, used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

