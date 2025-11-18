#ifndef __STATE_HPP___
#define __STATE_HPP___

#include <vector>

class State {

private:
    std::vector<int> tiles;
    int empty_cells;
    int side_length;
    void init(const std::vector<int>& tiles, int empty_cells, int side_length);
public:
    State(const std::vector<int>& tiles, int empty_cells);
    State(const std::vector<int>& tiles, int empty_cells, int side_length);
    ~State() = default;

    // Rule of five
    State(const State& other);
    State(const State&& other) noexcept;
    State& operator=(const State& other);
    State(State&& other) noexcept;
    State& operator=(State&& other) noexcept;

    int get_tile_row(int tile) const;
    int get_tile_column(int tile) const;

    int get_empty_cells() const;
    std::vector<int> get_empty_positions() const;
    std::vector<State> get_available_moves() const;
    bool operator==(const State &rhs) const;
};

#endif // __STATE_HPP___