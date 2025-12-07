/**
 * @file cuda_state_utils.hpp
 * @brief CUDA-compatible state utilities for GPU execution.
 *
 * This header provides device-side helper functions that work with DeviceState,
 * separate from the host-side State class which uses std::vector.
 */

#ifndef CUDA_STATE_UTILS_HPP
#define CUDA_STATE_UTILS_HPP

// Struct size must be deterministic for CUDA
// Max 4x4 board = 16 cells, so 16 ints is sufficient
struct DeviceState {
    int tiles[16];      // Tile positions (0-based linear indices)
    int num_tiles;      // Actual number of tiles
    int empty_cells;    // Number of empty cells
    int side_length;    // Board side length (typically 4)
    
    __device__ __host__ DeviceState() 
        : num_tiles(0), empty_cells(0), side_length(0) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) tiles[i] = -1;
    }
    
    /**
     * @brief Check equality of two device states.
     */
    __device__ __host__ bool operator==(const DeviceState& other) const {
        if (num_tiles != other.num_tiles || 
            empty_cells != other.empty_cells ||
            side_length != other.side_length) {
            return false;
        }
        #pragma unroll
        for (int i = 0; i < num_tiles; ++i) {
            if (tiles[i] != other.tiles[i]) return false;
        }
        return true;
    }
    
    /**
     * @brief Compute hash suitable for pheromone table indexing.
     * Uses XOR hash to combine tile positions.
     */
    __device__ __host__ size_t hash() const {
        size_t h = 0;
        #pragma unroll 16
        for (int i = 0; i < num_tiles; ++i) {
            h ^= tiles[i] + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
    
    /**
     * @brief Get row of tile at index i.
     */
    __device__ __host__ inline int get_tile_row(int i) const {
        return tiles[i] / side_length;
    }
    
    /**
     * @brief Get column of tile at index i.
     */
    __device__ __host__ inline int get_tile_column(int i) const {
        return tiles[i] % side_length;
    }
};

/**
 * @brief Get empty positions on device (max 16 positions for 4x4 board).
 * 
 * @param state Device state
 * @param empty_pos Output array (must have capacity for max empty cells)
 * @return Number of empty positions found
 */
__device__ __host__ inline int get_empty_positions_device(
    const DeviceState& state, 
    int* empty_pos
) {
    int num_cells = state.side_length * state.side_length;
    bool occupied[16];
    
    #pragma unroll
    for (int i = 0; i < 16; ++i) occupied[i] = false;
    
    #pragma unroll 16
    for (int i = 0; i < state.num_tiles; ++i) {
        if (state.tiles[i] >= 0 && state.tiles[i] < num_cells) {
            occupied[state.tiles[i]] = true;
        }
    }
    
    int count = 0;
    #pragma unroll 16
    for (int i = 0; i < num_cells; ++i) {
        if (!occupied[i]) {
            empty_pos[count++] = i;
        }
    }
    return count;
}

/**
 * @brief Manhattan distance heuristic on device.
 * 
 * @param state Current state
 * @param goal Goal state
 * @param weights Per-tile weights (length = num_tiles)
 * @return Weighted Manhattan distance
 */
__device__ __host__ inline int manhattan_distance_device(
    const DeviceState& state, 
    const DeviceState& goal,
    const int* weights
) {
    int dist = 0;
    
    #pragma unroll 16
    for (int i = 0; i < state.num_tiles; ++i) {
        int curr_pos = state.tiles[i];
        int goal_pos = goal.tiles[i];
        
        int curr_row = curr_pos / state.side_length;
        int curr_col = curr_pos % state.side_length;
        int goal_row = goal_pos / state.side_length;
        int goal_col = goal_pos % state.side_length;
        
        int tile_dist = abs(curr_row - goal_row) + abs(curr_col - goal_col);
        dist += tile_dist * weights[i];
    }
    
    return dist;
}

/**
 * @brief Generate available moves from a device state (max 4 moves per empty).
 * 
 * @param state Current state
 * @param moves Output array of successor states (preallocated)
 * @return Number of successor states generated
 */
__device__ __host__ inline int get_available_moves_device(
    const DeviceState& state, 
    DeviceState* moves
) {
    int empty_pos[16];
    int num_empty = get_empty_positions_device(state, empty_pos);
    int num_cells = state.side_length * state.side_length;
    int move_count = 0;
    
    // For each empty position
    for (int e = 0; e < num_empty; ++e) {
        int empty = empty_pos[e];
        
        // Check 4 directions: up, down, left, right
        int directions[4] = {
            -state.side_length,  // up
             state.side_length,  // down
            -1,                  // left
             1                   // right
        };
        
        for (int d = 0; d < 4; ++d) {
            int neighbor = empty + directions[d];
            
            // Boundary checks
            if (neighbor < 0 || neighbor >= num_cells) continue;
            
            // Boundary check for left/right moves
            if (directions[d] == -1 && empty % state.side_length == 0) continue;
            if (directions[d] == 1 && empty % state.side_length == state.side_length - 1) continue;
            
            // Create new state by swapping
            DeviceState new_state = state;
            bool swapped = false;
            
            #pragma unroll 16
            for (int i = 0; i < state.num_tiles; ++i) {
                if (new_state.tiles[i] == neighbor) {
                    new_state.tiles[i] = empty;
                    swapped = true;
                    break;
                }
            }
            
            if (swapped && move_count < 64) {  // Limit to avoid overflow
                moves[move_count++] = new_state;
            }
        }
    }
    
    return move_count;
}

#endif // CUDA_STATE_UTILS_HPP
