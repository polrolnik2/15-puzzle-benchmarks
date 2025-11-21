#ifndef __STATE_HPP___
#define __STATE_HPP___

#include <vector>

using namespace std;

class State {

private:
    vector<int> tiles;
    int empty_cells;
    int side_length;
    void init(const vector<int>& tiles, int empty_cells, int side_length);
public:
    State() = default;
    State(const vector<int>& tiles, int empty_cells);
    State(const vector<int>& tiles, int empty_cells, int side_length);
    ~State() = default;
    size_t hash() const;

    // Rule of five
    State(const State& other) = default;
    State& operator=(const State& other) = default;
    State(State&& other) = default;
    State& operator=(State&& other) = default;

    int get_tile_row(int tile) const;
    int get_tile_column(int tile) const;

    int get_empty_cells() const;
    vector<int> get_empty_positions() const;
    vector<State> get_available_moves() const;
    bool operator==(const State &rhs) const;
};

#endif // __STATE_HPP___