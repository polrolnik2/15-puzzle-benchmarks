#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <string>

#include "../A-star-model/state.hpp"
#include "../A-star-model/15-puzzle-a-star-solver.hpp"

using namespace std;

static vector<int> make_random_tiles(int side_size, int empty_cells, std::mt19937 &rng) {
    if (side_size != 4) throw runtime_error("Only side_size==4 is supported by this benchmark");
    int total = side_size * side_size;
    int ntiles = total - empty_cells;
    vector<int> positions(total);
    iota(positions.begin(), positions.end(), 0);
    shuffle(positions.begin(), positions.end(), rng);
    vector<int> tiles(ntiles);
    for (int i = 0; i < ntiles; ++i) tiles[i] = positions[i];
    return tiles;
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
    State goal_state(goal_tiles, empty_cells);

    PuzzleAStarSolver solver(20000);

    // CSV header
    cout << "side_size,empty_cells,instance_id,seed,time_ms,found,path_length" << '\n';

    vector<int> tiles = make_random_tiles(side_size, empty_cells, rng);
    State start_state(tiles, empty_cells);

    auto t0 = chrono::steady_clock::now();
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

    vector<State> path = solver.solve(start_state, goal_state, std::move(weights));
    auto t1 = chrono::steady_clock::now();
    double ms = chrono::duration_cast<chrono::duration<double, milli>>(t1 - t0).count();

    bool found = !path.empty();
    size_t plen = path.size();

    cout << side_size << ", empty cells: " << empty_cells << ", seed: " << seed << ", time: " << ms << "ms, solution found: " << (found?1:0) << ", steps: " << plen << '\n';

    return 0;
}
