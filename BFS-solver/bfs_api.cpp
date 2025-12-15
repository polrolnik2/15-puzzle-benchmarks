#include <chrono>
#include <vector>
#include <string>
#include "state.hpp"
#include "state_file_operations.hpp"
#include "15-puzzle-bfs-solver.hpp"

extern "C" {
    // Run BFS on a given instance file; returns time in milliseconds.
    // Returns 1 if solution found, 0 otherwise. visited_nodes and steps are outputs.
    int bfs_run_instance(
        const char* input_file,
        int side_size,
        int empty_cells,
        double* out_time_ms,
        int* out_steps,
        int* out_visited
    ) {
        if (!input_file || !out_time_ms || !out_steps || !out_visited) {
            return -1;
        }
        State start_state = read_state_from_file(std::string(input_file));
        int total = side_size * side_size;
        int ntiles = total - empty_cells;
        std::vector<int> goal_tiles(ntiles);
        for (int i = 0; i < ntiles; ++i) goal_tiles[i] = i;
        State goal_state(goal_tiles, empty_cells);

        auto t0 = std::chrono::steady_clock::now();
        int visited_nodes = 0;
        std::vector<State> path = BFSPuzzleSolver(start_state, goal_state, &visited_nodes);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

        *out_time_ms = ms;
        *out_steps = static_cast<int>(path.size());
        *out_visited = visited_nodes;
        return path.empty() ? 0 : 1;
    }
}
