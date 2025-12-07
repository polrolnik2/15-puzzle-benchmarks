#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>

#include "state.hpp"
#include "state_file_operations.hpp"
#include "generate_sample_state.hpp"

using namespace std;

int main(int argc, char** argv) {
    int side_size;
    int empty_cells;
    int depth;
    int seed;  
    string output_file;

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--side" && i + 1 < argc) { side_size = stoi(argv[++i]); }
        else if (a == "--empty" && i + 1 < argc) { empty_cells = stoi(argv[++i]); }
        else if (a == "--depth" && i + 1 < argc) { depth = stoi(argv[++i]); }     
        else if (a == "--seed" && i + 1 < argc) { seed = stoi(argv[++i]); }         
        else if (a == "--output-file" && i + 1 < argc) { output_file = argv[++i]; }
        else if (a == "--help") {
            cout << "Usage: benchmark-a-star [--side N] [--empty K] [--seed S]\n";
            return 0;
        }
    }
    mt19937 rng(seed);
    State sample = random_state_bfs(side_size, empty_cells, depth, rng);
    write_state_to_file(sample, output_file);
    return 0;
}
