#ifndef __GENERATE_SAMPLE_STATE_HPP___
#define __GENERATE_SAMPLE_STATE_HPP___

#include "state.hpp"
#include <random>
#include <queue>
#include <set>
#include <vector>
#include <stdexcept>

/**
 * @file generate_sample_state.hpp
 * @brief Utilities to create random puzzle states for benchmarks and testing.
 *
 * Two sampling strategies are provided:
 * - random walk: perform `target_depth` random legal moves from the goal state
 * - BFS sampling: collect all states at exact depth and pick one uniformly
 */

/**
 * @brief Generate a random state by performing a random walk from the solved state.
 *
 * The function starts from the canonical solved state (tiles in order) and
 * applies `target_depth` uniformly-random legal moves, returning the final state.
 *
 * @param side_size Board side length (e.g. 4 for 15-puzzle).
 * @param empty_cells Number of empty cells on the board.
 * @param target_depth Number of random moves to perform.
 * @param rng Random number generator to use (std::mt19937).
 * @return A sampled `State`.
 */
State random_state_random_walk(int side_size, int empty_cells, int target_depth, std::mt19937 &rng) {
    int total = side_size * side_size;
    int ntiles = total - empty_cells;
    vector<int> positions(ntiles);
    for (int i = 0; i < ntiles; ++i) positions[i] = i;
    State start_state(positions, empty_cells);
    State temp_state = start_state;
    for (int i = 0; i < target_depth; ++i) {
        auto moves = temp_state.get_available_moves();
        if (moves.empty()) break;
        std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
        temp_state = moves[dist(rng)];
    }
    return temp_state;
}

/**
 * @brief Generate a random state by uniform sampling among states at exact BFS depth.
 *
 * The function performs a breadth-first search from the solved state up to
 * `target_depth` and uniformly selects one of the states at that depth.
 * This is useful when you want a uniformly-distributed state at a given depth
 * rather than a biased random-walk sample.
 *
 * @param side_size Board side length (e.g. 4 for 15-puzzle).
 * @param empty_cells Number of empty cells on the board.
 * @param target_depth Depth to sample at (distance from solved state).
 * @param rng Random number generator to use (std::mt19937).
 * @return A sampled `State`. Returns the start (solved) state if no state exists at that depth.
 */
State random_state_bfs(int side_size, int empty_cells, int target_depth, std::mt19937 &rng) {
    int total = side_size * side_size;
    int ntiles = total - empty_cells;
    vector<int> positions;
    for (int i = 0; i < ntiles; ++i) positions.push_back(i);
    State start_state(positions, empty_cells);

    std::queue<std::pair<State, int>> frontier;
    std::set<State> explored;
    frontier.push({start_state, 0});
    explored.insert(start_state);

    std::vector<State> candidates;

    while (!frontier.empty()) {
        auto current = frontier.front();
        frontier.pop();
        State state = current.first;
        int depth = current.second;

        if (depth > target_depth) break;

        if (depth == target_depth) {
            candidates.push_back(state);
            continue;
        }

        auto moves = state.get_available_moves();
        for (const auto &move : moves) {
            if (explored.find(move) == explored.end()) {
                explored.insert(move);
                frontier.push({move, depth + 1});
            }
        }
    }

    if (candidates.empty()) {
        // no node found at that depth; return start as fallback
        return start_state;
    }

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng)];
}

#endif // __GENERATE_SAMPLE_STATE_HPP___