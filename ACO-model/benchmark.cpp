#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#include "state.hpp"
#include "state_file_operations.hpp"
#include "15-puzzle-aco-solver.hpp"

using namespace std;

/**
 * @file benchmark.cpp
 * @brief Benchmark runner for the CUDA ACO puzzle solver.
 *
 * This program loads a start state and runs the ACO solver with
 * configurable parameters, timing execution and printing a summary.
 */

int main(int argc, char** argv) {
    int side_size = 4;
    int empty_cells = 1;
    string input_file;
    string weights_raw;
    vector<int> weights;
    
    // ACO parameters
    ACOParams aco_params;
    
    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--side" && i + 1 < argc) { side_size = stoi(argv[++i]); }
        else if (a == "--empty" && i + 1 < argc) { empty_cells = stoi(argv[++i]); }
        else if (a == "--weights" && i + 1 < argc) { weights_raw = argv[++i]; }
        else if (a == "--input-file" && i + 1 < argc) { input_file = argv[++i]; }
        else if (a == "--num-ants" && i + 1 < argc) { aco_params.num_ants = stoi(argv[++i]); }
        else if (a == "--iterations" && i + 1 < argc) { aco_params.max_iterations = stoi(argv[++i]); }
        else if (a == "--max-steps" && i + 1 < argc) { aco_params.max_steps_per_ant = stoi(argv[++i]); }
        else if (a == "--alpha" && i + 1 < argc) { aco_params.alpha = stof(argv[++i]); }
        else if (a == "--beta" && i + 1 < argc) { aco_params.beta = stof(argv[++i]); }
        else if (a == "--evaporation" && i + 1 < argc) { aco_params.evaporation_rate = stof(argv[++i]); }
        else if (a == "--help") {
            cout << "Usage: benchmark_aco [options]\n";
            cout << "Options:\n";
            cout << "  --side N              Board size (default: 4)\n";
            cout << "  --empty K             Number of empty cells (default: 1)\n";
            cout << "  --input-file FILE     Input state file\n";
            cout << "  --weights W1,W2,...   Comma-separated tile weights\n";
            cout << "  --num-ants N          Number of ants per iteration (default: 256)\n";
            cout << "  --iterations N        Max ACO iterations (default: 100)\n";
            cout << "  --max-steps N         Max steps per ant (default: 100)\n";
            cout << "  --alpha A             Pheromone importance (default: 1.0)\n";
            cout << "  --beta B              Heuristic importance (default: 2.0)\n";
            cout << "  --evaporation E       Evaporation rate 0-1 (default: 0.1)\n";
            return 0;
        }
    }
    
    State start_state = read_state_from_file(input_file);
    
    // Parse weights
    std::stringstream ss(weights_raw);
    std::string token;
    while (getline(ss, token, ',')) {
        weights.push_back(stoi(token));
    }
    
    // Prepare goal state: tiles [0,1,2,...,ntiles-1]
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
    
    // Print configuration
    cout << "ACO Configuration:\n";
    cout << "  Ants: " << aco_params.num_ants << "\n";
    cout << "  Iterations: " << aco_params.max_iterations << "\n";
    cout << "  Alpha: " << aco_params.alpha << ", Beta: " << aco_params.beta << "\n";
    cout << "  Evaporation: " << aco_params.evaporation_rate << "\n\n";
    
    auto t0 = chrono::steady_clock::now();
    int visited_nodes = 0;
    vector<State> path = PuzzleSolveACO(start_state, goal_state, weights, aco_params, &visited_nodes);
    auto t1 = chrono::steady_clock::now();
    double ms = chrono::duration_cast<chrono::duration<double, milli>>(t1 - t0).count();
    
    bool found = !path.empty();
    size_t plen = found ? path.size() - 1 : 0; // Number of moves (not states)
    
    cout << "Results:\n";
    cout << "  Side: " << side_size << ", Empty cells: " << empty_cells << "\n";
    cout << "  Time: " << ms << " ms\n";
    cout << "  Solution found: " << (found ? "YES" : "NO") << "\n";
    cout << "  Path length: " << plen << " steps\n";
    cout << "  Total nodes visited: " << visited_nodes << "\n";
    
    return 0;
}
