/**
 * @file cuda_distance.hpp
 * @brief CUDA-compatible distance heuristics for DeviceState.
 * 
 * Provides device-compatible versions of distance functions that work
 * with DeviceState (from cuda_state_utils.hpp).
 */

#ifndef CUDA_DISTANCE_HPP
#define CUDA_DISTANCE_HPP

#include "cuda_state_utils.hpp"

/**
 * @brief Compute weighted Manhattan distance on device (GPU-compatible).
 * 
 * This is the device-side implementation of the distance heuristic.
 * Use this in CUDA kernels; for host code, use the manhattan_distance()
 * function from distance.hpp.
 *
 * @param state Current device state
 * @param goal_state Goal device state
 * @param weights Per-tile weights (size = state.num_tiles)
 * @return Weighted Manhattan distance (sum of weighted tile distances)
 */
__device__ __host__ inline int manhattan_distance_device(
    const DeviceState& state, 
    const DeviceState& goal_state, 
    const int* weights
) {
    int dist = 0;
    
    // Unroll for better GPU performance
    #pragma unroll 16
    for (int i = 0; i < state.num_tiles; ++i) {
        int curr_pos = state.tiles[i];
        int goal_pos = goal_state.tiles[i];
        
        // Convert linear index to row/col
        int curr_row = curr_pos / state.side_length;
        int curr_col = curr_pos % state.side_length;
        int goal_row = goal_pos / state.side_length;
        int goal_col = goal_pos % state.side_length;
        
        // Manhattan distance for this tile
        int tile_dist = abs(curr_row - goal_row) + abs(curr_col - goal_col);
        dist += tile_dist * weights[i];
    }
    
    return dist;
}

/**
 * @brief Compute unweighted Manhattan distance on device.
 * 
 * Simpler version that assumes all tiles have equal weight (1).
 *
 * @param state Current device state
 * @param goal_state Goal device state
 * @return Manhattan distance (sum of tile distances)
 */
__device__ __host__ inline int manhattan_distance_simple_device(
    const DeviceState& state, 
    const DeviceState& goal_state
) {
    int dist = 0;
    
    #pragma unroll 16
    for (int i = 0; i < state.num_tiles; ++i) {
        int curr_pos = state.tiles[i];
        int goal_pos = goal_state.tiles[i];
        
        int curr_row = curr_pos / state.side_length;
        int curr_col = curr_pos % state.side_length;
        int goal_row = goal_pos / state.side_length;
        int goal_col = goal_pos % state.side_length;
        
        dist += abs(curr_row - goal_row) + abs(curr_col - goal_col);
    }
    
    return dist;
}

/**
 * @brief Compute linear conflict heuristic on device (more accurate but slower).
 * 
 * Linear conflict: if two tiles are in the same row/column but inverted
 * order, add 2 to the heuristic.
 *
 * @param state Current device state
 * @param goal_state Goal device state
 * @return Manhattan distance + 2 × (number of linear conflicts)
 */
__device__ __host__ inline int linear_conflict_device(
    const DeviceState& state, 
    const DeviceState& goal_state
) {
    int dist = manhattan_distance_simple_device(state, goal_state);
    int conflicts = 0;
    
    // Check row conflicts
    for (int row = 0; row < state.side_length; ++row) {
        int row_base = row * state.side_length;
        
        for (int i = 0; i < state.num_tiles; ++i) {
            int pos_i = state.tiles[i];
            int goal_pos_i = goal_state.tiles[i];
            
            // Check if both tiles are in same row in current and goal
            if (pos_i / state.side_length != row) continue;
            if (goal_pos_i / state.side_length != row) continue;
            
            // Check for conflicts with subsequent tiles
            for (int j = i + 1; j < state.num_tiles; ++j) {
                int pos_j = state.tiles[j];
                int goal_pos_j = goal_state.tiles[j];
                
                if (pos_j / state.side_length != row) continue;
                if (goal_pos_j / state.side_length != row) continue;
                
                // Conflict if order is inverted
                if ((pos_i < pos_j) != (goal_pos_i < goal_pos_j)) {
                    conflicts += 2;
                }
            }
        }
    }
    
    return dist + conflicts;
}

#endif // CUDA_DISTANCE_HPP
