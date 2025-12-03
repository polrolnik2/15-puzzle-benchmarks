#ifndef PUZZLE_ASTAR_SOLVER_HPP
#define PUZZLE_ASTAR_SOLVER_HPP

/**
 * @file 15-puzzle-a-star-solver.hpp
 * @brief A* solver adapter for the N-puzzle `State` type.
 */

/**
 * @brief Solve the puzzle using an A* implementation.
 *
 * @param start Starting puzzle state.
 * @param goal Goal puzzle state.
 * @param weights Per-tile movement weights used by the heuristic.
 * @param visited_nodes Optional out-parameter to receive number of visited nodes.
 * @return Sequence of states from start to goal (empty if no solution found).
 */
std::vector<State> PuzzleSolveAstar(const State &start, const State &goal, std::vector<int> weights, int* visited_nodes = nullptr);

#endif // PUZZLE_ASTAR_SOLVER_HPP
