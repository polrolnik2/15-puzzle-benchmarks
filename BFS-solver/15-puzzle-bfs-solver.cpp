#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include <queue>
#include <set>
#include "state.hpp"

#include "15-puzzle-bfs-solver.hpp"

std::vector<State> BFSPuzzleSolver(const State &start, const State &goal, int* visited_nodes) {
    std::vector<State> path;
    // BFS implementation to find the path from start to goal
    std::queue<std::pair<State, std::vector<State>>> frontier;
    std::set<State> explored;
    frontier.push({start, {start}});
    while (!frontier.empty()) {
        auto current = frontier.front();
        frontier.pop();
        State state = current.first;
        std::vector<State> current_path = current.second;

        if (state == goal) {
            path = current_path;
            break;
        }

        explored.insert(state);
        if (visited_nodes) {
            (*visited_nodes)++;
        }
        auto moves = state.get_available_moves();
        for (const auto &move : moves) {
            if (explored.find(move) == explored.end()) {
                std::vector<State> new_path = current_path;
                new_path.push_back(move);
                frontier.push({move, new_path});
            }
        }
    }
    return path;
}