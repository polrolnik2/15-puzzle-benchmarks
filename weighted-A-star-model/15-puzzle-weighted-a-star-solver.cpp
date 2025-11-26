#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include "state.hpp"
#include "distance.hpp"
#include "15-puzzle-weighted-a-star-solver.hpp"
#include "../dependencies/a-star/cpp/stlastar.h"

float ASTAR_HEURISTIC_WEIGHT = 1.0; // Default weight

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

// Heuristic: sum of Manhattan distances of matching tile indices
float PuzzleAStarState::GoalDistanceEstimate(PuzzleAStarState &nodeGoal) {
    return ASTAR_HEURISTIC_WEIGHT * static_cast<float>(manhattan_distance(this->to_state(), nodeGoal.to_state(), this->weights_));
}

bool PuzzleAStarState::IsGoal(PuzzleAStarState &nodeGoal) {
    return IsSameState(nodeGoal);
}

bool PuzzleAStarState::GetSuccessors(AStarSearch<PuzzleAStarState> *astarsearch, PuzzleAStarState * parent_node) {
    auto moves = this->to_state().get_available_moves();
    for (auto &mv : moves) {
        PuzzleAStarState tmp(mv, this->weights_);
        if (!astarsearch->AddSuccessor(tmp)) return false;
    }
    return true;
}

float PuzzleAStarState::GetCost(PuzzleAStarState & successor) {
    return static_cast<float>(manhattan_distance(this->to_state(), successor.to_state(), this->weights_));
}

bool PuzzleAStarState::IsSameState(PuzzleAStarState &rhs) {
    return this->puzzle_state_ == rhs.puzzle_state_;
}

size_t PuzzleAStarState::Hash() {
    // Simple rolling hash over internal tile positions
    size_t h = 1469598103934665603ULL; // FNV offset
    h ^= static_cast<size_t>(this->puzzle_state_.get_empty_cells() + 1);
    h *= 1099511628211ULL;
    h ^= this->puzzle_state_.hash();
    return h;
}

State PuzzleAStarState::to_state() const {
    return puzzle_state_;
}

std::vector<State> PuzzleSolveAstar(const State &start, const State &goal, std::vector<int> weights, float heuristic_weight, int* visited_nodes) {
    ASTAR_HEURISTIC_WEIGHT = heuristic_weight;
    PuzzleAStarState sstart(start, weights);
    PuzzleAStarState sgoal(goal, weights);

    AStarSearch<PuzzleAStarState> search_;
    search_.SetStartAndGoalStates(sstart, sgoal);

    unsigned int result = 0;
    do {
        result = search_.SearchStep();
    } while (result == AStarSearch<PuzzleAStarState>::SEARCH_STATE_SEARCHING);

    std::vector<State> path;
    if (result == AStarSearch<PuzzleAStarState>::SEARCH_STATE_SUCCEEDED) {
        PuzzleAStarState *p = search_.GetSolutionStart();
        while (p) {
            path.push_back(p->to_state());
            p = search_.GetSolutionNext();
        }
        p = search_.GetSolutionEnd();
        path.push_back(p->to_state());
    }

    if (visited_nodes) {
        *visited_nodes = search_.GetStepCount();
    }

    search_.FreeSolutionNodes();
    return path;
}
