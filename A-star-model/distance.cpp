#include <cstdlib>

#include "state.hpp"
#include "distance.hpp"

using namespace std;

int manhattan_distance(const State& state, const State& goal_state, vector<int>weights) {
    int distance = 0;
    if (weights.size() != static_cast<size_t>(16 - state.get_empty_cells())) {
        throw invalid_argument("Weights size does not match number of tiles in state");
    }
    for (int tile = 0; tile < 16 - state.get_empty_cells(); ++tile) {
        int current_row = state.get_tile_row(tile);
        int current_col = state.get_tile_column(tile);
        int goal_row = goal_state.get_tile_row(tile);
        int goal_col = goal_state.get_tile_column(tile);
        distance += weights[tile] * (abs(current_row - goal_row) + abs(current_col - goal_col));
    }
    return distance;
}