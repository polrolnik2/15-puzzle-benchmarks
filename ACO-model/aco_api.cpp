#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#include "state.hpp"
#include "state_file_operations.hpp"
#include "15-puzzle-aco-solver.hpp"

/**
 * @file aco_api.cpp
 * @brief C-friendly wrapper for the CUDA ACO solver so it can be called from Python via ctypes.
 */

extern "C" {
    int aco_run_instance(
        const char* input_file,
        int side_size,
        int empty_cells,
        int max_iterations,
        int num_ants,
        int max_steps_per_ant,
        float alpha,
        float beta,
        float evaporation_rate,
        float pheromone_deposit,
        float initial_pheromone,
        float exploitation_prob,
        float local_evaporation,
        const int* weights,
        int weights_len,
        double* out_time_ms,
        int* out_steps,
        int* out_visited,
        int * out_distance
    ) {
        if (!input_file || !out_time_ms || !out_steps || !out_visited) {
            return -1;
        }
        if (side_size <= 0 || empty_cells <= 0 || max_iterations <= 0 || num_ants <= 0 || max_steps_per_ant <= 0) {
            return -1;
        }

        State start_state = read_state_from_file(std::string(input_file));
        int total = side_size * side_size;
        int ntiles = total - empty_cells;

        // Goal tiles are 0..ntiles-1
        std::vector<int> goal_tiles(ntiles);
        for (int i = 0; i < ntiles; ++i) goal_tiles[i] = i;
        State goal_state(goal_tiles, empty_cells);

        // Prepare weights; default to 1s if not provided or too short
        std::vector<int> weights_vec(ntiles, 1);
        if (weights && weights_len >= ntiles) {
            weights_vec.assign(weights, weights + ntiles);
        }

        ACOParams params;
        params.max_iterations = max_iterations;
        params.num_ants = num_ants;
        params.max_steps_per_ant = max_steps_per_ant;
        params.alpha = alpha;
        params.beta = beta;
        params.evaporation_rate = evaporation_rate;
        params.pheromone_deposit = pheromone_deposit;
        params.initial_pheromone = initial_pheromone;
        params.exploitation_prob = exploitation_prob;
        params.local_evaporation = local_evaporation;

        auto t0 = std::chrono::steady_clock::now();
        int visited_nodes = 0;
        int distance = 0;
        std::vector<State> path = PuzzleSolveACO(start_state, goal_state, weights_vec, params, &visited_nodes, &distance);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

        *out_distance = distance;
        *out_time_ms = ms;
        *out_steps = static_cast<int>(path.size()); // consistent with A* and BFS: return number of states in path
        *out_visited = visited_nodes;

        return path.empty() ? 0 : 1;
    }
}
