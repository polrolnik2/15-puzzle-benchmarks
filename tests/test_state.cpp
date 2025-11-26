// Google Test for State (creation and position getters)
#include <gtest/gtest.h>
#include <stdexcept>

#include "state.hpp"

TEST(StateTest, ValidCreationAndPosition) {
    int empty_cells = 1;
    int n = 16 - empty_cells; // number of tiles present
    std::vector<int> tiles(n);
    for (int i = 0; i < n; ++i) {
        tiles[i] = i; // place tile i at position i
    }

    State s(tiles, empty_cells);

    EXPECT_EQ(s.get_tile_row(0), 0);
    EXPECT_EQ(s.get_tile_column(0), 0);

    // position 5 -> row 1, column 1
    EXPECT_EQ(s.get_tile_row(5), 1);
    EXPECT_EQ(s.get_tile_column(5), 1);

    // last tile index (n-1) -> row 3, column 2 when n==15
    EXPECT_EQ(s.get_tile_row(n-1), (n-1) / 4);
    EXPECT_EQ(s.get_tile_column(n-1), (n-1) % 4);
}

TEST(StateTest, DuplicateTilesThrows) {
    int empty_cells = 1;
    int n = 16 - empty_cells;
    std::vector<int> tiles(n);
    for (int i = 0; i < n; ++i) tiles[i] = i;
    // create a duplicate position: tiles[3] == tiles[2]
    tiles[3] = tiles[2];

    EXPECT_THROW(State(tiles, empty_cells), std::invalid_argument);
}

TEST(StateTest, OutOfRangeTileValueThrows) {
    int empty_cells = 1;
    int n = 16 - empty_cells;
    std::vector<int> tiles(n);
    for (int i = 0; i < n; ++i) tiles[i] = i;
    // set an out-of-range position
    tiles[4] = 100;

    EXPECT_THROW(State(tiles, empty_cells), std::invalid_argument);
    }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
