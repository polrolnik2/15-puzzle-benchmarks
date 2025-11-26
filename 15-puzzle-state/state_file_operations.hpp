#ifndef __STATE_FILE_OPERATIONS_HPP___
#define __STATE_FILE_OPERATIONS_HPP___

State read_state_from_file(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    int side_length, empty_cells;
    infile >> side_length >> empty_cells;
    int n = side_length * side_length - empty_cells;
    std::vector<int> tiles(n);
    for (int i = 0; i < n; ++i) {
        infile >> tiles[i];
    }
    return State(tiles, empty_cells);
}

void write_state_to_file(const State& state, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    int side_length = 4; // assuming standard 15-puzzle
    int empty_cells = state.get_empty_cells();
    outfile << side_length << " " << empty_cells << "\n";
    int n = side_length * side_length - empty_cells;
    for (int i = 0; i < n; ++i) {
        outfile << state.get_tile_row(i) * side_length + state.get_tile_column(i) << " ";
    }
    outfile << "\n";
}

#endif // __STATE_FILE_OPERATIONS_HPP___