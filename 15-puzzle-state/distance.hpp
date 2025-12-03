#ifndef __DISTANCE_HPP___
#define __DISTANCE_HPP___

/**
 * @file distance.hpp
 * @brief Heuristic distance functions for N-puzzle states.
 */

/**
 * @brief Compute weighted Manhattan distance between two states.
 *
 * @param state Current state.
 * @param goal_state Goal state to compare against.
 * @param weights Per-tile weights (length equals number of tiles).
 * @return Sum of weighted Manhattan distances.
 */
int manhattan_distance(const State& state, const State& goal_state, std::vector<int>weights);

#endif // __DISTANCE_HPP___