#include "15-puzzle-a-star-solver.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include "distance.hpp"

// Heuristic: sum of Manhattan distances of matching tile indices
float PuzzleAStarState::GoalDistanceEstimate(PuzzleAStarState &nodeGoal) {
    return static_cast<float>(manhattan_distance(this->to_state(), nodeGoal.to_state(), this->weights_));
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
    return this->empty_cells_ == rhs.empty_cells_ && this->tiles_ == rhs.tiles_;
}

size_t PuzzleAStarState::Hash() {
    // Simple rolling hash over internal tile positions
    size_t h = 1469598103934665603ULL; // FNV offset
    h ^= static_cast<size_t>(this->empty_cells_ + 1);
    h *= 1099511628211ULL;
    for (int v : tiles_) {
        h ^= static_cast<size_t>(v + 1);
        h *= 1099511628211ULL; // FNV prime
    }
    return h;
}

State PuzzleAStarState::to_state() const {
    return State(this->tiles_, this->empty_cells_);
}

PuzzleAStarSolver::PuzzleAStarSolver(int maxNodes)
    : search_(maxNodes)
{
}

std::vector<State> PuzzleAStarSolver::solve(const State &start, const State &goal, std::vector<int> weights) {
    PuzzleAStarState sstart(start, weights);
    PuzzleAStarState sgoal(goal, weights);

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

    search_.FreeSolutionNodes();
    return path;
}
