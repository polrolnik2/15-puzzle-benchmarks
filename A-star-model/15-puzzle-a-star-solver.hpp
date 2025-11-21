#ifndef PUZZLE_ASTAR_SOLVER_HPP
#define PUZZLE_ASTAR_SOLVER_HPP

#include "state.hpp"
#include "../dependencies/a-star/cpp/stlastar.h"
#include <vector>
#include <cstddef>
#include <memory>

// Puzzle state type that implements the A* user-state interface
class PuzzleAStarState {
public:
    PuzzleAStarState() = default;
    PuzzleAStarState(const State &s, const std::vector<int>& weights) {
        puzzle_state_ = s;
        weights_ = weights;
    }
    ~PuzzleAStarState() = default;

    // AStarState interface
    float GoalDistanceEstimate(PuzzleAStarState &nodeGoal);
    bool IsGoal(PuzzleAStarState &nodeGoal);
    bool GetSuccessors(AStarSearch<PuzzleAStarState> *astarsearch, PuzzleAStarState *parent_node);
    float GetCost(PuzzleAStarState &successor);
    bool IsSameState(PuzzleAStarState &rhs);
    size_t Hash();

    // Convert to a runtime-owned State object (caller takes ownership via State semantics)
    State to_state() const;

private:
    std::vector<int> weights_;
    State puzzle_state_;
};

// Simple solver wrapper that uses the A* engine on PuzzleAStarState
class PuzzleAStarSolver {
public:
    PuzzleAStarSolver(int maxNodes = 10000);

    // Solve from start -> goal. Returns sequence of States from start to goal (inclusive).
    std::vector<State> solve(const State &start, const State &goal, std::vector<int> weights);

private:
    AStarSearch<PuzzleAStarState> search_;
};

#endif // PUZZLE_ASTAR_SOLVER_HPP
