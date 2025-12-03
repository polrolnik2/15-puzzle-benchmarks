/**
 * @file state.hpp
 * @brief N-puzzle state representation (tiles and empty cell handling).
 *
 * This header declares the State class used across solvers and tools.
 */

#ifndef __STATE_HPP___
#define __STATE_HPP___

using namespace std;

/**
 * @brief Represents a board state for sliding puzzles (e.g. 15-puzzle).
 *
 * The class stores the tile cell indices, the number of empty cells, and
 * the board size (side_length). It exposes helpers to query tile positions
 * and to generate legal successor states.
 */
class State {

private:
    vector<int> tiles;
    int empty_cells;
    int side_length;
    void init(const vector<int>& tiles, int empty_cells, int side_length);
public:
    State() = default;
    /**
     * @brief Construct a State using inferred side length (square) from tile count.
     *
     * @param tiles Tile values in row-major order (length = side*side - empty_cells).
     * @param empty_cells Number of empty cells.
     * @throws std::invalid_argument on inconsistent input.
     */
    State(const vector<int>& tiles, int empty_cells);

    /**
     * @brief Construct a State with explicit side length.
     *
     * @param tiles Tile values in row-major order.
     * @param empty_cells Number of empty cells.
     * @param side_length Board side length.
     */
    State(const vector<int>& tiles, int empty_cells, int side_length);
    ~State() = default;

    /**
     * @brief Compute a stable hash for this state.
     *
     * The hash is suitable for use in unordered containers.
     * @return A size_t hash value.
     */
    size_t hash() const;

    // Rule of five
    State(const State& other) = default;
    State& operator=(const State& other) = default;
    State(State&& other) = default;
    State& operator=(State&& other) = default;

    /**
     * @brief Return the row index (0-based) of the given tile index in the stored tile ordering.
     *
     * @param tile Index of the tile in the internal ordering (0..ntiles-1).
     * @return Row index of the tile (0-based).
     */
    int get_tile_row(int tile) const;

    /**
     * @brief Return the column index (0-based) of the given tile index.
     *
     * @param tile Index of the tile in the internal ordering (0..ntiles-1).
     * @return Column index of the tile (0-based).
     */
    int get_tile_column(int tile) const;

    /**
     * @brief Number of empty cells on the board.
     * @return Number of empty cells.
     */
    int get_empty_cells() const;

    /**
     * @brief Return the linear indices of empty cell positions (in row-major order).
     *
     * @return Vector of positions (each value in 0..side_length*side_length-1).
     */
    vector<int> get_empty_positions() const;

    /**
     * @brief Generate all legal successor states from this state.
     *
     * Successors are returned in an unspecified order. Each successor represents
     * a single legal sliding move of one tile into an adjacent empty cell.
     *
     * @return Vector of successor `State` instances.
     */
    vector<State> get_available_moves() const;

    /**
     * @brief Equality comparison between two states (same tile layout and empties).
     */
    bool operator==(const State &rhs) const;

    /**
     * @brief Strict weak ordering used for ordered containers (std::set).
     */
    bool operator<(const State &rhs) const;
};

#endif // __STATE_HPP___