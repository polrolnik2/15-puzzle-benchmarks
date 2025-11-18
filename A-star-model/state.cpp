#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "state.hpp"

using namespace std;

void State::init(const vector<int>& tiles, int empty_cells, int side_length) {
    if (tiles.empty()) {
        throw invalid_argument("Tiles array cannot be empty");
    }
    int num_cells = side_length * side_length;
    if (empty_cells < 0 || empty_cells > num_cells) {
        throw invalid_argument("empty_cells out of range");
    }
    int n = num_cells - empty_cells;
    for (int i = 0; i < n; ++i) {
        if (tiles[i] < 0 || tiles[i] >= num_cells) {
            throw invalid_argument("Tile values must be in range [0,15]");
        }
        for (int j = 0; j < i; ++j) {
            if (tiles[j] == tiles[i]) {
                throw invalid_argument("Duplicate tile positions");
            }
        }
    }
    this->side_length = side_length;
    this->tiles = tiles;
    this->empty_cells = empty_cells;
}

State::State(const vector<int>& tiles, int empty_cells, int side_length) {
    init(tiles, empty_cells, side_length);
}

State::State(const vector<int>& tiles, int empty_cells) {
    init(tiles, empty_cells, 4);
}

// Copy constructor (deep copy)
State::State(const State& other) {
    int num_cells = other.side_length * other.side_length;
    empty_cells = other.empty_cells;
    int n = num_cells - empty_cells;
    tiles = other.tiles;
    side_length = other.side_length;
}

State::State(State&& other) noexcept {
    tiles = std::move(other.tiles);
    empty_cells = other.empty_cells;
    side_length = other.side_length;
    other.empty_cells = 0;
}

State& State::operator=(const State& other) {
    if (this == &other) return *this;
    empty_cells = other.empty_cells;
    int n = other.side_length * other.side_length - empty_cells;
    tiles = other.tiles;
    side_length = other.side_length;
    return *this;
}

State& State::operator=(State&& other) noexcept {
    if (this == &other) return *this;
    tiles = std::move(other.tiles);
    empty_cells = other.empty_cells;
    side_length = other.side_length;
    other.empty_cells = 0;
    return *this;
}

int State::get_tile_row(int tile) const {
    return tiles[tile] / side_length;
}

int State::get_tile_column(int tile) const {
    return tiles[tile] % side_length;
}

int State::get_empty_cells() const {
    return empty_cells;
}

vector<int> State::get_empty_positions() const {
    vector<int> empty_positions;
    int num_cells = side_length * side_length;
    vector<bool> occupied(num_cells, false);
    for (int i = 0; i < num_cells - empty_cells; ++i) {
        if (!tiles.empty()) {
            int pos = tiles[i];
            if (pos >= 0 && pos < num_cells) occupied[pos] = true;
        }
    }
    for (int i = 0; i < num_cells; ++i) {
        if (!occupied[i]) empty_positions.push_back(i);
    }
    return empty_positions;
}

vector<State> State::get_available_moves() const {
    vector<State> moves;
    vector<int> empty_positions = get_empty_positions();
    int num_cells = side_length * side_length;
    for (int empty_pos : empty_positions) {
        // Check possible moves (up, down, left, right)
        vector<int> directions = {-side_length, side_length, -1, 1};
        for (int dir : directions) {
            int neighbor_pos = empty_pos + dir;
            if (neighbor_pos >= 0 && neighbor_pos < num_cells) {
                // Create new tiles array for the new state
                std::vector<int> new_tiles = tiles;
                copy(tiles.begin(), tiles.end(), new_tiles.begin());
                // Find the tile that is at neighbor_pos and swap it with empty_pos
                bool swapped = false;
                for (int i = 0; i < num_cells - empty_cells; ++i) {
                    if (new_tiles[i] == neighbor_pos) {
                        new_tiles[i] = empty_pos;
                        swapped = true;
                        break;
                    }
                }
                if (swapped) {
                    State new_state(new_tiles, empty_cells);
                    moves.push_back(new_state);
                }
            }
        }
    }
    return moves;
}

bool State::operator==(const State &rhs) const {
    if (side_length != rhs.side_length) return false;
    if (empty_cells != rhs.empty_cells) return false;
    int n = side_length * side_length - empty_cells;
    for (int i = 0; i < n; ++i) {
        if (tiles[i] != rhs.tiles[i]) return false;
    }
    return true;
}