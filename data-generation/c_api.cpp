#include <string>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "state.hpp"
#include "state_file_operations.hpp"
#include "generate_sample_state.hpp"

namespace fs = std::filesystem;

extern "C" {
    // Generate a random-walk instance and write it to a file inside out_dir.
    // Returns 0 on success, negative on error. On success, writes the full
    // path into out_path_buf (NUL-terminated) if buffer is large enough.
    int datagen_random_walk_to_file(
        int side_size,
        int empty_cells,
        int depth,
        unsigned int seed,
        const char* out_dir,
        char* out_path_buf,
        int out_path_buf_len
    ) {
        if (!out_dir || !out_path_buf || out_path_buf_len <= 0) return -1;
        try {
            // ensure directory exists
            fs::create_directories(out_dir);

            // build unique filename
            auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::string fname = "random_walk_d" + std::to_string(depth) + "_" + std::to_string(seed) + "_" + std::to_string(now) + ".state";
            fs::path full = fs::path(out_dir) / fname;

            // generate state
            std::mt19937 rng(seed);
            State s = random_state_random_walk(side_size, empty_cells, depth, rng);
            write_state_to_file(s, full.string());

            // copy path to buffer
            std::string p = full.string();
            if ((int)p.size() + 1 > out_path_buf_len) return -2;
            std::memcpy(out_path_buf, p.c_str(), p.size() + 1);
            return 0;
        } catch (...) {
            return -3;
        }
    }
}
