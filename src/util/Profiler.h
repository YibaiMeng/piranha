#pragma once
#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

class Profiler {
public:
    Profiler();

    void start();
    void clear();
    void accumulate(std::string tag);
    double get_elapsed(std::string tag);
    double get_elapsed_all();
    void dump_all();

    void track_alloc(size_t bytes, int gpu_id);
    void track_free(size_t bytes, int gpu_id);
    void tag_mem();
    void dump_mem_tags();
    std::vector<double> get_max_mem_mb();
    
    void add_comm_round();
    void dump_comm_rounds();

    void add_comm_bytes(size_t bytes, bool tx);
    void add_intergpu_comm_bytes(size_t bytes, int src_id, int dst_id);
    size_t get_comm_tx_bytes();
    size_t get_comm_rx_bytes();
    void dump_comm_bytes();
    void print_intergpu_comm_bytes();
    std::map<std::string, double> accumulators;

private:

    bool running;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    double total;

    double mem_mb[10];
    std::vector<std::pair<double, double> > tags;
    double max_mem_mb[10];

    size_t rounds;

    size_t bytes_tx;
    size_t bytes_rx;

    // TODO: not hardcode max number of GPU per party;
    // intergpu_bytes[i][j] means the number of bytes sent from GPU i to GPU j;
    size_t intergpu_bytes[8][8];
};

// TODO global list of profilers


