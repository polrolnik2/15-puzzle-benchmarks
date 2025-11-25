#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>

#include "state.hpp"
#include "15-puzzle-a-star-solver.hpp"
#include "15-puzzle-bfs-solver.hpp"

using namespace std;

static State random_state(int side_size, int empty_cells, std::mt19937 &rng) {
    if (side_size != 4) throw runtime_error("Only side_size==4 is supported by this benchmark");
    int total = side_size * side_size;
    int ntiles = total - empty_cells;
    vector<int> positions(ntiles);
    for (int i = 0; i < ntiles; ++i) positions[i] = i;
    State start_state(positions, empty_cells);
    std::uniform_int_distribution<int> dist(0, 50);
    int random_value = dist(rng);
    State temp_state = start_state;
    for (int i = 0; i < random_value; ++i) {
        auto moves = temp_state.get_available_moves();
        temp_state = moves[rng() % moves.size()];
    }
    return temp_state;
}

int main(int argc, char** argv) {
    int side_size = 4;
    int empty_cells = 1;
    unsigned int seed = (unsigned int)chrono::high_resolution_clock::now().time_since_epoch().count();

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--side" && i + 1 < argc) { side_size = stoi(argv[++i]); }
        else if (a == "--empty" && i + 1 < argc) { empty_cells = stoi(argv[++i]); }
        else if (a == "--seed" && i + 1 < argc) { seed = (unsigned int)stoul(argv[++i]); }
        else if (a == "--help") {
            cout << "Usage: benchmark-a-star [--side N] [--empty K] [--seed S]\n";
            return 0;
        }
    }

    if (empty_cells < 0 || empty_cells > side_size * side_size) {
        cerr << "empty_cells must be in [0," << side_size * side_size << "]\n";
        return 3;
    }

    mt19937 rng(seed);

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

    // CSV header
    cout << "side_size,empty_cells,instance_id,seed,time_ms,found,path_length" << '\n';

    State start_state;
    try {
        start_state = random_state(side_size, empty_cells, rng);
    } catch (const std::exception& e) {
        cerr << "Error generating random state: " << e.what() << '\n';
        return 4;
    }

    int nweights = ntiles;
    vector<int> weights(nweights);
    uniform_int_distribution<int> wdist(1, 10);
    for (int i = 0; i < nweights; ++i) weights[i] = wdist(rng);

    // print generated weights (optional)
    cout << "weights:";
    for (size_t i = 0; i < weights.size(); ++i) {
        if (i) cout << ',';
        cout << weights[i];
    }
    cout << '\n';

    auto t0 = chrono::steady_clock::now();
    // vector<State> path = PuzzleSolveAstar(start_state, goal_state, weights);
    vector<State> path = BFSPuzzleSolver(start_state, goal_state);
    auto t1 = chrono::steady_clock::now();
    double ms = chrono::duration_cast<chrono::duration<double, milli>>(t1 - t0).count();

    bool found = !path.empty();
    size_t plen = path.size();

    cout << side_size << ", empty cells: " << empty_cells << ", seed: " << seed << ", time: " << ms << "ms, solution found: " << (found?1:0) << ", steps: " << plen << '\n';

    return 0;
}
