#include "15-puzzle-aco-solver.hpp"
#include "cuda_state_utils.hpp"
#include "cuda_distance.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

/**
 * @file 15-puzzle-aco-solver.cu
 * @brief CUDA kernels and host code for ACO-based puzzle solving.
 * 
 * Uses DeviceState from cuda_state_utils.hpp and cuda_distance.hpp
 * for all GPU operations.
 */

// Convert host State to DeviceState
__host__ DeviceState to_device_state(const State& s) {
    DeviceState ds;
    ds.empty_cells = s.get_empty_cells();
    ds.side_length = 4; // Assuming 4x4 for 15-puzzle
    ds.num_tiles = ds.side_length * ds.side_length - ds.empty_cells;
    for (int i = 0; i < ds.num_tiles; ++i) {
        int row = s.get_tile_row(i);
        int col = s.get_tile_column(i);
        ds.tiles[i] = row * ds.side_length + col;
    }
    return ds;
}

// Convert DeviceState back to host State
__host__ State to_host_state(const DeviceState& ds) {
    std::vector<int> tiles_vec(ds.num_tiles);
    for (int i = 0; i < ds.num_tiles; ++i) {
        tiles_vec[i] = ds.tiles[i];
    }
    return State(tiles_vec, ds.empty_cells, ds.side_length);
}

// Get empty positions on device
__device__ int get_empty_positions_device(const DeviceState& state, int* empty_pos) {
    int num_cells = state.side_length * state.side_length;
    bool occupied[16] = {false};
    
    for (int i = 0; i < state.num_tiles; ++i) {
        if (state.tiles[i] >= 0 && state.tiles[i] < num_cells) {
            occupied[state.tiles[i]] = true;
        }
    }
    
    int count = 0;
    for (int i = 0; i < num_cells; ++i) {
        if (!occupied[i]) {
            empty_pos[count++] = i;
        }
    }
    return count;
}

// Generate available moves on device
__device__ int get_available_moves_device(const DeviceState& state, DeviceState* moves) {
    int empty_pos[16];
    int num_empty = get_empty_positions_device(state, empty_pos);
    int num_cells = state.side_length * state.side_length;
    int move_count = 0;
    
    for (int e = 0; e < num_empty; ++e) {
        int empty = empty_pos[e];
        int directions[4] = {-state.side_length, state.side_length, -1, 1};
        
        for (int d = 0; d < 4; ++d) {
            int neighbor = empty + directions[d];
            
            // Boundary checks
            if (neighbor < 0 || neighbor >= num_cells) continue;
            if (directions[d] == -1 && empty % state.side_length == 0) continue;
            if (directions[d] == 1 && empty % state.side_length == state.side_length - 1) continue;
            
            // Create new state
            DeviceState new_state = state;
            for (int i = 0; i < state.num_tiles; ++i) {
                if (new_state.tiles[i] == neighbor) {
                    new_state.tiles[i] = empty;
                    moves[move_count++] = new_state;
                    break;
                }
            }
        }
    }
    return move_count;
}
// Note: manhattan_distance_device is now defined in cuda_distance.hpp
// included at the top of this file
}

// CUDA kernel: Each ant constructs a path
__global__ void aco_construct_solutions_kernel(
    DeviceState start_state,
    DeviceState goal_state,
    int* d_weights,
    float* d_pheromones,
    int pheromone_size,
    ACOParams params,
    DeviceState* d_ant_paths,      // [num_ants][max_steps_per_ant]
    int* d_ant_path_lengths,       // [num_ants]
    int* d_ant_found_goal,         // [num_ants]
    unsigned long long seed
) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= params.num_ants) return;
    
    // Initialize random state
    curandState rand_state;
    curand_init(seed, ant_id, 0, &rand_state);
    
    DeviceState current = start_state;
    int path_len = 0;
    d_ant_paths[ant_id * params.max_steps_per_ant] = current;
    d_ant_found_goal[ant_id] = 0;
    
    for (int step = 0; step < params.max_steps_per_ant; ++step) {
        // Check if goal reached
        if (current == goal_state) {
            d_ant_found_goal[ant_id] = 1;
            d_ant_path_lengths[ant_id] = path_len;
            return;
        }
        
        // Get available moves
        DeviceState moves[64];
        int num_moves = get_available_moves_device(current, moves);
        
        if (num_moves == 0) break;
        
        // Calculate probabilities based on pheromones and heuristic
        float probabilities[64];
        float total_prob = 0.0f;
        
        for (int i = 0; i < num_moves; ++i) {
            size_t hash_idx = moves[i].hash() % pheromone_size;
            float pheromone = d_pheromones[hash_idx];
            float heuristic = 1.0f / (1.0f + manhattan_distance_device(moves[i], goal_state, d_weights));
            
            probabilities[i] = powf(pheromone, params.alpha) * powf(heuristic, params.beta);
            total_prob += probabilities[i];
        }
        
        // Select next move using roulette wheel selection
        float rand_val = curand_uniform(&rand_state) * total_prob;
        float cumulative = 0.0f;
        int selected = 0;
        
        for (int i = 0; i < num_moves; ++i) {
            cumulative += probabilities[i];
            if (rand_val <= cumulative) {
                selected = i;
                break;
            }
        }
        
        current = moves[selected];
        path_len++;
        d_ant_paths[ant_id * params.max_steps_per_ant + path_len] = current;
    }
    
    d_ant_path_lengths[ant_id] = path_len;
}

// CUDA kernel: Evaporate pheromones
__global__ void evaporate_pheromones_kernel(float* d_pheromones, int size, float evaporation_rate, float min_pheromone) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_pheromones[idx] = fmaxf(d_pheromones[idx] * (1.0f - evaporation_rate), min_pheromone);
    }
}

// CUDA kernel: Deposit pheromones for successful ants
__global__ void deposit_pheromones_kernel(
    DeviceState* d_ant_paths,
    int* d_ant_path_lengths,
    int* d_ant_found_goal,
    float* d_pheromones,
    int pheromone_size,
    int num_ants,
    int max_steps,
    float deposit_amount
) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= num_ants) return;
    
    if (d_ant_found_goal[ant_id] == 1) {
        int path_len = d_ant_path_lengths[ant_id];
        float quality = deposit_amount / (1.0f + path_len);
        
        for (int i = 0; i < path_len; ++i) {
            DeviceState state = d_ant_paths[ant_id * max_steps + i];
            size_t hash_idx = state.hash() % pheromone_size;
            atomicAdd(&d_pheromones[hash_idx], quality);
        }
    }
}

// Host function
std::vector<State> PuzzleSolveACO(
    const State &start, 
    const State &goal, 
    std::vector<int> weights,
    const ACOParams& params,
    int* visited_nodes
) {
    // Convert to device states
    DeviceState d_start = to_device_state(start);
    DeviceState d_goal = to_device_state(goal);
    
    // Allocate device memory
    int pheromone_table_size = 100000; // Hash table size
    float* d_pheromones;
    int* d_weights;
    DeviceState* d_ant_paths;
    int* d_ant_path_lengths;
    int* d_ant_found_goal;
    
    cudaMalloc(&d_pheromones, pheromone_table_size * sizeof(float));
    cudaMalloc(&d_weights, weights.size() * sizeof(int));
    cudaMalloc(&d_ant_paths, params.num_ants * params.max_steps_per_ant * sizeof(DeviceState));
    cudaMalloc(&d_ant_path_lengths, params.num_ants * sizeof(int));
    cudaMalloc(&d_ant_found_goal, params.num_ants * sizeof(int));
    
    // Initialize pheromones
    std::vector<float> init_pheromones(pheromone_table_size, params.initial_pheromone);
    cudaMemcpy(d_pheromones, init_pheromones.data(), pheromone_table_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // Best solution tracking
    std::vector<State> best_solution;
    int best_length = INT_MAX;
    int total_visited = 0;
    
    // ACO main loop
    int threads_per_block = 256;
    int blocks = (params.num_ants + threads_per_block - 1) / threads_per_block;
    
    for (int iter = 0; iter < params.max_iterations; ++iter) {
        // Construct solutions
        aco_construct_solutions_kernel<<<blocks, threads_per_block>>>(
            d_start, d_goal, d_weights, d_pheromones, pheromone_table_size,
            params, d_ant_paths, d_ant_path_lengths, d_ant_found_goal,
            (unsigned long long)(iter * 12345)
        );
        cudaDeviceSynchronize();
        
        // Copy results back
        std::vector<int> h_path_lengths(params.num_ants);
        std::vector<int> h_found_goal(params.num_ants);
        cudaMemcpy(h_path_lengths.data(), d_ant_path_lengths, params.num_ants * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_found_goal.data(), d_ant_found_goal, params.num_ants * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Find best ant in this iteration
        for (int ant = 0; ant < params.num_ants; ++ant) {
            total_visited += h_path_lengths[ant];
            if (h_found_goal[ant] == 1 && h_path_lengths[ant] < best_length) {
                best_length = h_path_lengths[ant];
                
                // Copy best path
                std::vector<DeviceState> h_path(best_length + 1);
                cudaMemcpy(h_path.data(), 
                          &d_ant_paths[ant * params.max_steps_per_ant],
                          (best_length + 1) * sizeof(DeviceState),
                          cudaMemcpyDeviceToHost);
                
                best_solution.clear();
                for (int i = 0; i <= best_length; ++i) {
                    best_solution.push_back(to_host_state(h_path[i]));
                }
            }
        }
        
        // Evaporate pheromones
        int evap_blocks = (pheromone_table_size + threads_per_block - 1) / threads_per_block;
        evaporate_pheromones_kernel<<<evap_blocks, threads_per_block>>>(
            d_pheromones, pheromone_table_size, params.evaporation_rate, 0.01f
        );
        
        // Deposit pheromones
        deposit_pheromones_kernel<<<blocks, threads_per_block>>>(
            d_ant_paths, d_ant_path_lengths, d_ant_found_goal,
            d_pheromones, pheromone_table_size, params.num_ants,
            params.max_steps_per_ant, params.pheromone_deposit
        );
        cudaDeviceSynchronize();
        
        // Early termination if good solution found
        if (!best_solution.empty() && best_length < 50) break;
    }
    
    // Cleanup
    cudaFree(d_pheromones);
    cudaFree(d_weights);
    cudaFree(d_ant_paths);
    cudaFree(d_ant_path_lengths);
    cudaFree(d_ant_found_goal);
    
    if (visited_nodes) *visited_nodes = total_visited;
    
    return best_solution;
}
