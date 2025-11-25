#ifndef PUZZLE_ASTAR_SOLVER_HPP
#define PUZZLE_ASTAR_SOLVER_HPP

std::vector<State> PuzzleSolveAstar(const State &start, const State &goal, std::vector<int> weights, int* visited_nodes = nullptr);

#endif // PUZZLE_ASTAR_SOLVER_HPP
