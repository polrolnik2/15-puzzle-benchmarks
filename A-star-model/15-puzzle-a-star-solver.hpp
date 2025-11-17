#ifndef PUZZLE_ASTAR_SOLVER_HPP
#define PUZZLE_ASTAR_SOLVER_HPP

#include "state.hpp"
#include "../dependencies/a-star/cpp/stlastar.h"
#include <vector>
#include <cstddef>

// Puzzle state type that implements the A* user-state interface
class PuzzleAStarState : public AStarState<PuzzleAStarState> {
public:
    PuzzleAStarState() = default;
    PuzzleAStarState(const State &s) : s_(s) {}

    // AStarState interface
    float GoalDistanceEstimate(PuzzleAStarState &nodeGoal) override;
    bool IsGoal(PuzzleAStarState &nodeGoal) override;
    bool GetSuccessors(AStarSearch<PuzzleAStarState> *astarsearch, PuzzleAStarState *parent_node) override;
    float GetCost(PuzzleAStarState &successor) override;
    bool IsSameState(PuzzleAStarState &rhs) override;
    size_t Hash() override;

    const State& state() const { return s_; }

private:
    State s_ { nullptr, 0 };
};

// Simple solver wrapper that uses the A* engine on PuzzleAStarState
class PuzzleAStarSolver {
public:
    PuzzleAStarSolver(int maxNodes = 10000);

    // Solve from start -> goal. Returns sequence of States from start to goal (inclusive).
    std::vector<State> solve(const State &start, const State &goal);

private:
    AStarSearch<PuzzleAStarState> search_;
};

#endif // PUZZLE_ASTAR_SOLVER_HPP
