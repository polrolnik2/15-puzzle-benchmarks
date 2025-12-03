#ifndef PUZZLE_ASTAR_SOLVER_HPP
#define PUZZLE_ASTAR_SOLVER_HPP

/**
 * @file 15-puzzle-weighted-a-star-solver.hpp
 * @brief Weighted A* solver adapter for N-puzzle `State`.
 */

/**
 * @brief Solve the puzzle using weighted A*.
 *
 * @param start Starting puzzle state.
 * @param goal Goal puzzle state.
 * @param weights Per-tile weights used by the heuristic.
 * @param heuristic_weight Weight applied to the heuristic component (w*A*).
 * @param visited_nodes Optional out-parameter to receive number of visited nodes.
 * @return Sequence of states from start to goal (empty if no solution found).
 */
std::vector<State> PuzzleSolveWeightedAstar(const State &start, const State &goal, std::vector<int> weights, float heuristic_weight = 1.0, int* visited_nodes = nullptr);

#endif // PUZZLE_ASTAR_SOLVER_HPP
