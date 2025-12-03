#ifndef __15_PUZZLE_BFS_SOLVER_HPP___
#define __15_PUZZLE_BFS_SOLVER_HPP___

/**
 * @file 15-puzzle-bfs-solver.hpp
 * @brief Breadth-first search solver adapter for N-puzzle `State`.
 */

/**
 * @brief Solve the puzzle using BFS (useful for small depths).
 *
 * @param start Starting puzzle state.
 * @param goal Goal puzzle state.
 * @param visited_nodes Optional out-parameter to receive number of visited nodes.
 * @return Sequence of states from start to goal (empty if no solution found).
 */
std::vector<State> BFSPuzzleSolver(const State &start, const State &goal, int* visited_nodes = nullptr);

#endif // __15_PUZZLE_BFS_SOLVER_HPP___