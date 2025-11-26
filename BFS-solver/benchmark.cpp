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
#include "15-puzzle-bfs-solver.hpp"

using namespace std;

int main(int argc, char** argv) {
    int side_size;
    int empty_cells;
    string input_file;
    string weights_raw;
    vector<int> weights;

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--side" && i + 1 < argc) { side_size = stoi(argv[++i]); }
        else if (a == "--empty" && i + 1 < argc) { empty_cells = stoi(argv[++i]); }
        else if (a == "--weights" && i + 1 < argc) { weights_raw = argv[++i]; }
        else if (a == "--input-file" && i + 1 < argc) { input_file = argv[++i]; }
        else if (a == "--help") {
            cout << "Usage: benchmark-a-star [--side N] [--empty K] [--seed S]\n";
            return 0;
        }
    }
    State start_state = read_state_from_file(input_file);

    std::stringstream ss(weights_raw);
    std::string token;
    while (getline(ss, token, ',')) {
        weights.push_back(stoi(token));
    }
    // Prepare a simple goal state: tiles [0,1,2,...,ntiles-1]
    int total = side_size * side_size;
    int ntiles = total - empty_cells;
    vector<int> goal_tiles(ntiles);
    for (int i = 0; i < ntiles; ++i) goal_tiles[i] = i;
    State goal_state;
    try {
        goal_state = State(goal_tiles, empty_cells);
    } catch (const std::exception& e) {
        cerr << "Error creating goal state: " << e.what() << '\n';
        return 2;
    }

    // print generated weights (optional)
    cout << "weights:";
    for (size_t i = 0; i < weights.size(); ++i) {
        if (i) cout << ',';
        cout << weights[i];
    }
    cout << '\n';

    auto t0 = chrono::steady_clock::now();
    int visited_nodes = 0;
    vector<State> path = BFSPuzzleSolver(start_state, goal_state, &visited_nodes);
    auto t1 = chrono::steady_clock::now();
    double ms = chrono::duration_cast<chrono::duration<double, milli>>(t1 - t0).count();

    bool found = !path.empty();
    size_t plen = path.size();

    cout << side_size << ", empty cells: " << empty_cells << ", time: " << ms << "ms, solution found: " << (found?1:0) << ", steps: " << plen << ", visited nodes: " << visited_nodes << '\n';

    return 0;
}
