// Google Test for State::get_available_moves
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <set>

#include "../A-star-model/state.hpp"

static int* make_tiles_from_empty_positions(const std::vector<int>& empties) {
    int empty_cells = static_cast<int>(empties.size());
    int n = 16 - empty_cells;
    std::vector<int> pos;
    pos.reserve(n);
    for (int i = 0; i < 16; ++i) {
        if (std::find(empties.begin(), empties.end(), i) == empties.end()) pos.push_back(i);
    }
    // allocate array expected by State constructor
    int* tiles = new int[n];
    for (int i = 0; i < n; ++i) tiles[i] = pos[i];
    return tiles;
}

static std::set<int> collect_swapped_empty_positions(const State& s) {
    auto original = s.get_empty_positions();
    auto moves = s.get_available_moves();
    std::set<int> result;
    std::set<int> origset(original.begin(), original.end());
    for (const auto &mv : moves) {
        auto epos = mv.get_empty_positions();
        if (epos != original) {
            for (int e : epos) if (origset.find(e) == origset.end()) result.insert(e);
        }
    }
    return result;
}

TEST(AvailableMoves, SingleEmptyCorner) {
    std::vector<int> empties = {15};
    int* tiles = make_tiles_from_empty_positions(empties);
    State s(tiles, static_cast<int>(empties.size()));

    auto found = collect_swapped_empty_positions(s);
    std::set<int> expected = {11, 14};
    EXPECT_EQ(found, expected);
}

TEST(AvailableMoves, SingleEmptyEdge) {
    std::vector<int> empties = {4};
    int* tiles = make_tiles_from_empty_positions(empties);
    State s(tiles, static_cast<int>(empties.size()));

    // neighbors: up(0), down(8), left(3), right(5)
    std::set<int> expected = {0, 8, 3, 5};
    auto found = collect_swapped_empty_positions(s);
    EXPECT_EQ(found, expected);
}

TEST(AvailableMoves, SingleEmptyMiddle) {
    std::vector<int> empties = {5};
    int* tiles = make_tiles_from_empty_positions(empties);
    State s(tiles, static_cast<int>(empties.size()));

    // neighbors: up(1), down(9), left(4), right(6)
    std::set<int> expected = {1, 9, 4, 6};
    auto found = collect_swapped_empty_positions(s);
    EXPECT_EQ(found, expected);
}

TEST(AvailableMoves, TwoEmptyCells) {
    std::vector<int> empties = {14, 15};
    int* tiles = make_tiles_from_empty_positions(empties);
    State s(tiles, static_cast<int>(empties.size()));

    // For empties 14 and 15, valid moves are tiles adjacent to these that are not empty:
    // 14 neighbors: 10, 13, 15 -> 15 is empty so exclude -> {10,13}
    // 15 neighbors: 11, 14 -> 14 is empty so exclude -> {11}
    static std::set<int> expected = {10, 11, 13};
    auto found = collect_swapped_empty_positions(s);
    EXPECT_EQ(found, expected);
}


