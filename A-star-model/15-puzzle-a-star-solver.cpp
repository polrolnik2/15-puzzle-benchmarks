#include "15-puzzle-a-star-solver.hpp"
#include <algorithm>
#include <functional>
#include <numeric>

// Heuristic: sum of Manhattan distances of matching tile indices
float PuzzleAStarState::GoalDistanceEstimate(PuzzleAStarState &nodeGoal) {
    const State &a = this->s_;
    const State &b = nodeGoal.s_;
    int na = 16 - a.get_empty_cells();
    int nb = 16 - b.get_empty_cells();
    int n = std::min(na, nb);
    float total = 0.0f;
    for (int tile = 0; tile < n; ++tile) {
        int ar = a.get_tile_row(tile);
        int ac = a.get_tile_column(tile);
        int br = b.get_tile_row(tile);
        int bc = b.get_tile_column(tile);
        total += static_cast<float>(std::abs(ar - br) + std::abs(ac - bc));
    }
    return total;
}

bool PuzzleAStarState::IsGoal(PuzzleAStarState &nodeGoal) {
    return IsSameState(nodeGoal);
}

bool PuzzleAStarState::GetSuccessors(AStarSearch<PuzzleAStarState> *astarsearch, PuzzleAStarState * /*parent_node*/) {
    auto moves = s_.get_available_moves();
    for (auto &mv : moves) {
        PuzzleAStarState tmp(mv);
        if (!astarsearch->AddSuccessor(tmp)) return false;
    }
    return true;
}

float PuzzleAStarState::GetCost(PuzzleAStarState &/*successor*/) {
    return 1.0f; // each slide costs 1
}

bool PuzzleAStarState::IsSameState(PuzzleAStarState &rhs) {
    const State &a = this->s_;
    const State &b = rhs.s_;
    int na = 16 - a.get_empty_cells();
    int nb = 16 - b.get_empty_cells();
    if (na != nb) return false;
    for (int tile = 0; tile < na; ++tile) {
        if (a.get_tile_row(tile) != b.get_tile_row(tile) || a.get_tile_column(tile) != b.get_tile_column(tile)) return false;
    }
    return true;
}

size_t PuzzleAStarState::Hash() {
    const State &a = this->s_;
    int n = 16 - a.get_empty_cells();
    // Simple rolling hash over tile positions
    size_t h = 1469598103934665603ULL; // FNV offset
    for (int tile = 0; tile < n; ++tile) {
        int pos = a.get_tile_row(tile) * 4 + a.get_tile_column(tile);
        h ^= static_cast<size_t>(pos + 1);
        h *= 1099511628211ULL; // FNV prime
    }
    return h;
}

PuzzleAStarSolver::PuzzleAStarSolver(int maxNodes)
    : search_(maxNodes)
{
}

std::vector<State> PuzzleAStarSolver::solve(const State &start, const State &goal) {
    PuzzleAStarState sstart(start);
    PuzzleAStarState sgoal(goal);

    search_.SetStartAndGoalStates(sstart, sgoal);

    unsigned int result = 0;
    do {
        result = search_.SearchStep();
    } while (result == AStarSearch<PuzzleAStarState>::SEARCH_STATE_SEARCHING);

    std::vector<State> path;
    if (result == AStarSearch<PuzzleAStarState>::SEARCH_STATE_SUCCEEDED) {
        PuzzleAStarState *p = search_.GetSolutionStart();
        while (p) {
            path.push_back(p->state());
            p = search_.GetSolutionNext();
        }
    }

    search_.FreeSolutionNodes();
    return path;
}
