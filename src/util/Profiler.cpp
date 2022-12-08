#include "util/Profiler.h"
#include <iostream>
#include <sstream>
#include <loguru.hpp>

Profiler::Profiler() : running(false), total(0),  rounds(0), bytes_tx(0), bytes_rx(0) {
    for(int i = 0; i < 8; i++) {
       for(int j = 0; j < 8; j++) {
            intergpu_bytes[i][j] = 0;
        }
    }
    for(int i = 0; i < 10; i++) {
        max_mem_mb[i] = 0;
        mem_mb[i] = 0;
    }
}

void Profiler::start() {
    running = true;
    start_time = std::chrono::system_clock::now();
}

void Profiler::clear() {
    running = false;
    total = 0;
    accumulators.clear();
    for(int i = 0; i < 10; i++)  {
        mem_mb[i] = 0.0;
        max_mem_mb[i] = 0.0;
    }
    tags.clear();

    rounds = 0;

    bytes_tx = 0;
    bytes_rx = 0;

    for(int i = 0; i < 8; i++) {
       for(int j = 0; j < 8; j++) {
            intergpu_bytes[i][j] = 0;
        }
    }

}

void Profiler::accumulate(std::string tag) {
    if (running) {
        running = false;

        double us_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - start_time
        ).count();

        accumulators[tag] += us_elapsed / 1000.0;
        total += us_elapsed / 1000.0;
    }
}

double Profiler::get_elapsed(std::string tag) {
    return accumulators[tag]; 
}

double Profiler::get_elapsed_all() {
    return total;
}

void Profiler::dump_all() {
    std::cout << "Total: " << total << " ms" << std::endl;
    for (auto &s : accumulators) {
        std::cout << "  " << s.first << ": " << s.second << " ms" << std::endl;
    }
    std::cout << std::endl << "-------------------" << std::endl;
}

void Profiler::track_alloc(size_t bytes, int gpu_id) {
    if (!running) return;
    int curr_device;
    mem_mb[gpu_id] += ((double)bytes) / 1024.0 / 1024.0;
    
    if(mem_mb[gpu_id] > max_mem_mb[gpu_id]) max_mem_mb[gpu_id] = mem_mb[gpu_id];
}

void Profiler::track_free(size_t bytes, int gpu_id) {
    if (!running) return;
    
    mem_mb[gpu_id]-= ((double)bytes) / 1024.0 / 1024.0;
}

void Profiler::tag_mem() {
    if (!running) return;

    double ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start_time
    ).count();

    //std::cout << mem << std::endl;
    tags.push_back(std::make_pair(ms_elapsed, mem_mb));
    //std::cout << "MEM," << ms_elapsed << "," << mem << std::endl;
}

void Profiler::dump_mem_tags() {
    std::cout << std::endl << "-------------------" << std::endl;
    for (auto &p : tags) {
        std::cout << p.first << "," << p.second << std::endl;
    }
    std::cout << std::endl << "-------------------" << std::endl;
}

std::vector<double> Profiler::get_max_mem_mb() {
    std::vector<double> res;
    for(int i = 0; i < 10; i++) res.push_back(max_mem_mb[i]);
    return res;
}

void Profiler::add_comm_round() {
    if (!running) return;

    rounds++;
}

void Profiler::dump_comm_rounds() {
    std::cout << std::endl << "-------------------" << std::endl;
    std::cout << "Communication rounds: " << rounds;
    std::cout << std::endl << "-------------------" << std::endl;
}

void Profiler::add_comm_bytes(size_t bytes, bool tx) {
    if (tx) {
        bytes_tx += bytes;
    } else {
        bytes_rx += bytes;
    }
}

void Profiler::add_intergpu_comm_bytes(size_t bytes, int src_id, int dst_id) {
    intergpu_bytes[src_id][dst_id] += bytes;
}



size_t Profiler::get_comm_tx_bytes() {
    return bytes_tx;
}

size_t Profiler::get_comm_rx_bytes() {
    return bytes_rx;
}

void Profiler::dump_comm_bytes() {
    std::cout << std::endl << "-------------------" << std::endl;
    std::cout << "Communication bytes | sent (MB): " << bytes_tx / (1024.0 * 1024.0) <<  " received (MB): " << bytes_rx / (1024.0 * 1024.0);
    std::cout << std::endl << "-------------------" << std::endl;
}

