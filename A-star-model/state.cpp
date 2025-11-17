#include <stdexcept>
#include <string>
#include <format>
#include <vector>
#include <algorithm>

#include "state.hpp"

State::State(int* tiles, int empty_cells) {
    if (tiles == nullptr) {
        throw std::invalid_argument("Tiles array cannot be null");
    }
    for (int i = 0; i < 16-empty_cells; ++i) {
        if (tiles[i] < 0 || tiles[i] >= 16) {
            throw std::invalid_argument("Tile values must be in range [0,15]");
        }
        for (int j = 0; j < i; ++j) {
            if (tiles[j] == tiles[i]) {
                throw std::invalid_argument(std::format("Tiles: {} and {} occupy the same position", i, j));
            }
        }
    }
    this->tiles = tiles;
    this->empty_cells = empty_cells;
}

State::~State() {
    delete[] tiles;
}

// Copy constructor (deep copy)
State::State(const State& other) {
    empty_cells = other.empty_cells;
    int n = 16 - empty_cells;
    if (other.tiles) {
        tiles = new int[n];
        std::copy(other.tiles, other.tiles + n, tiles);
    } else {
        tiles = nullptr;
    }
}

State& State::operator=(const State& other) {
    if (this == &other) return *this;
    delete[] tiles;
    empty_cells = other.empty_cells;
    int n = 16 - empty_cells;
    if (other.tiles) {
        tiles = new int[n];
        std::copy(other.tiles, other.tiles + n, tiles);
    } else {
        tiles = nullptr;
    }
    return *this;
}

// Move constructor
State::State(State&& other) noexcept {
    tiles = other.tiles;
    empty_cells = other.empty_cells;
    other.tiles = nullptr;
    other.empty_cells = 0;
}

State& State::operator=(State&& other) noexcept {
    if (this == &other) return *this;
    delete[] tiles;
    tiles = other.tiles;
    empty_cells = other.empty_cells;
    other.tiles = nullptr;
    other.empty_cells = 0;
    return *this;
}

int State::get_tile_row(int tile) const {
    return tiles[tile] / 4;
}

int State::get_tile_column(int tile) const {
    return tiles[tile] % 4;
}

int State::get_empty_cells() const {
    return empty_cells;
}

std::vector<int> State::get_empty_positions() const {
    std::vector<int> empty_positions;
    empty_positions.reserve(empty_cells);
    bool occupied[16] = {false};
    for (int i = 0; i < 16 - empty_cells; ++i) {
        if (tiles) {
            int pos = tiles[i];
            if (pos >= 0 && pos < 16) occupied[pos] = true;
        }
    }
    for (int i = 0; i < 16; ++i) {
        if (!occupied[i]) empty_positions.push_back(i);
    }
    return empty_positions;
}

std::vector<State> State::get_available_moves() const {
    std::vector<State> moves;
    std::vector<int> empty_positions = get_empty_positions();
    for (int empty_pos : empty_positions) {
        // Check possible moves (up, down, left, right)
        std::vector<int> directions = {-4, 4, -1, 1};
        for (int dir : directions) {
            int neighbor_pos = empty_pos + dir;
            if (neighbor_pos >= 0 && neighbor_pos < 16) {
                // Create new tiles array for the new state
                int* new_tiles = new int[16 - empty_cells];
                std::copy(tiles, tiles + (16 - empty_cells), new_tiles);
                // Find the tile that is at neighbor_pos and swap it with empty_pos
                bool swapped = false;
                for (int i = 0; i < 16 - empty_cells; ++i) {
                    if (new_tiles[i] == neighbor_pos) {
                        new_tiles[i] = empty_pos;
                        swapped = true;
                        break;
                    }
                }
                if (swapped) {
                    State new_state(new_tiles, empty_cells);
                    moves.push_back(new_state);
                } else {
                    delete[] new_tiles;
                }
            }
        }
    }
    return moves;
}