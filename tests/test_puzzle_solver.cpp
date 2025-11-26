// Google Test for PuzzleAStarSolver
#include <gtest/gtest.h>
#include "state.hpp"
#include "15-puzzle-a-star-solver.hpp"

TEST(PuzzleSolver, OneMoveSolution) {
    int empty_cells = 1;
    int n = 16 - empty_cells;
    std::vector<int> weights(n, 1); // uniform weights
    // start: tiles 0..14 at positions 0..14, empty at 15
    std::vector<int> start_tiles;
    for (int i = 0; i < n; ++i) start_tiles.push_back(i);
    State start(start_tiles, empty_cells);

    // goal: same but tile 14 moved from pos 14 -> pos 15 (empty at 14)
    std::vector<int> goal_tiles;
    for (int i = 0; i < n; ++i) goal_tiles.push_back(i);
    State goal(goal_tiles, empty_cells);

    auto path = PuzzleSolveAstar(start, goal, weights);

    // Expect at least two states: start and goal
    EXPECT_GE(path.size(), 2u);
    // first state equals start
    EXPECT_TRUE(path.front().get_empty_positions() == start.get_empty_positions());
    // last state equals goal (empty positions)
    EXPECT_TRUE(path.back().get_empty_positions() == goal.get_empty_positions());
}
