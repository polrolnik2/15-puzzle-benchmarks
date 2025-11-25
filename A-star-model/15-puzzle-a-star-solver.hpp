#ifndef PUZZLE_ASTAR_SOLVER_HPP
#define PUZZLE_ASTAR_SOLVER_HPP

#include <vector>
#include "state.hpp"

std::vector<State> PuzzleSolveAstar(const State &start, const State &goal, std::vector<int> weights);

#endif // PUZZLE_ASTAR_SOLVER_HPP
