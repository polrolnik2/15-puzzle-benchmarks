#include "15-puzzle-aco-solver.hpp"
#include "cuda_state_utils.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>

/**
 * @file 15-puzzle-aco-solver.cu
 * @brief CUDA kernels and host code for ACO-based puzzle solving.
 * 
 * Uses DeviceState from cuda_state_utils.hpp and cuda_distance.hpp
 * for all GPU operations.
 */

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Convert host State to DeviceState (HOST ONLY - uses std::vector)
DeviceState to_device_state(const State& s) {
    DeviceState ds;
    ds.empty_cells = s.get_empty_cells();
    ds.side_length = 4; // Assuming 4x4 for 15-puzzle
    ds.num_tiles = ds.side_length * ds.side_length - ds.empty_cells;
    // State stores tile values as linear indices directly
    // No conversion needed - just copy the values
    for (int i = 0; i < ds.num_tiles; ++i) {
        // Note: State class stores tiles as linear indices already
        // get_tile_row and get_tile_column extract row/col from the stored index
        // We need the stored index value, not row/col
        ds.tiles[i] = s.get_tile_row(i) * ds.side_length + s.get_tile_column(i);
    }
    return ds;
}

// Convert DeviceState back to host State (HOST ONLY - uses std::vector)
State to_host_state(const DeviceState& ds) {
    std::vector<int> tiles_vec(ds.num_tiles);
    for (int i = 0; i < ds.num_tiles; ++i) {
        tiles_vec[i] = ds.tiles[i];
    }
    return State(tiles_vec, ds.empty_cells, ds.side_length);
}

// Note: get_empty_positions_device and get_available_moves_device are defined in cuda_state_utils.hpp
// Note: manhattan_distance_device is now defined in cuda_distance.hpp
// All these functions are included at the top of this file

// CUDA kernel: Initialize random states for ants
__global__ void init_curand_kernel(curandState* rand_states, int num_ants, unsigned long long seed) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= num_ants) return;
    curand_init(seed, ant_id, 0, &rand_states[ant_id]);
}

// CUDA kernel: Each ant constructs a path
__global__ void aco_construct_solutions_kernel(
    DeviceState start_state,
    DeviceState goal_state,
    int* d_weights,
    float* d_pheromones,
    int pheromone_size,
    int num_ants,
    int max_steps_per_ant,
    float alpha,
    float beta,
    DeviceState* d_ant_paths,      // [num_ants][max_steps_per_ant]
    int* d_ant_path_lengths,       // [num_ants]
    int* d_ant_found_goal,         // [num_ants]
    int* d_ant_best_distances,     // [num_ants] track closest approach to goal
    curandState* rand_states       // Pre-initialized random states
) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= num_ants) return;    
    // Use pre-initialized random state - add bounds check
    if (ant_id >= num_ants) return;  // Double check
    curandState local_rand_state = rand_states[ant_id];
    
    DeviceState current = start_state;
    int path_len = 0;
    
    // Track the best (minimum) distance to goal this ant achieves
    int best_distance = manhattan_distance_device(start_state, goal_state, d_weights);
    d_ant_best_distances[ant_id] = best_distance;
    
    d_ant_paths[ant_id * max_steps_per_ant] = current;
    d_ant_found_goal[ant_id] = 0;
    
    for (int step = 0; step < max_steps_per_ant; ++step) {
        // Check if goal reached
        if (current == goal_state) {
            d_ant_found_goal[ant_id] = 1;
            d_ant_path_lengths[ant_id] = path_len;
            return;
        }
        
        // Get available moves - use static array (max 4 moves for 1 empty cell)
        DeviceState moves[4];
        int num_moves = get_available_moves_device(current, moves);

        // FAIL: Every state MUST have at least one available move
        if (num_moves <= 0 || num_moves > 4) {
            if (ant_id == 0) {
                printf("ERROR: Invalid move count! num_moves=%d\n", num_moves);
            }
            d_ant_found_goal[ant_id] = -5;
            d_ant_path_lengths[ant_id] = path_len;
            return;
        }
        
        // Calculate probabilities based on pheromones and heuristic
        float probabilities[4];
        float total_prob = 0.0f;
        
        for (int i = 0; i < num_moves; ++i) {
            size_t hash_idx = moves[i].hash() % pheromone_size;
            float pheromone = fmaxf(d_pheromones[hash_idx], 0.01f); 
            float heuristic = 1.0f / (1.0f + manhattan_distance_device(moves[i], goal_state, d_weights));
            
            // FAIL: Heuristic MUST be in valid range
            assert(heuristic > 0.0f && heuristic <= 1.0f && "ERROR: Invalid heuristic value!");
            
            probabilities[i] = powf(pheromone, alpha) * powf(heuristic, beta);
            
            // FAIL: Probability must be valid (not NaN or Inf)
            assert(probabilities[i] >= 0.0f && probabilities[i] < 1e9f && "ERROR: Invalid probability!");
            
            total_prob += probabilities[i];
        }
        
        // FAIL: Total probability MUST be valid
        assert(total_prob > 0.0f && total_prob < 1e9f && "ERROR: Invalid total probability!");
        
        // Select next move using roulette wheel selection
        float rand_val = curand_uniform(&local_rand_state) * total_prob;
        float cumulative = 0.0f;
        int selected = 0;
        
        for (int i = 0; i < num_moves; ++i) {
            cumulative += probabilities[i];
            if (rand_val <= cumulative) {
                selected = i;
                break;
            }
        }
        
        // FAIL: Selected move MUST be valid
        assert(selected >= 0 && selected < num_moves && "ERROR: Invalid move selection!");
        
        current = moves[selected];
        path_len++;
        
        // Update best distance if this state is closer to goal
        int current_distance = manhattan_distance_device(current, goal_state, d_weights);
        if (current_distance < best_distance) {
            best_distance = current_distance;
            d_ant_best_distances[ant_id] = best_distance;
        }
        
        // FAIL: Path length MUST not exceed buffer
        if (path_len >= max_steps_per_ant) {
            d_ant_found_goal[ant_id] = -6;
            d_ant_path_lengths[ant_id] = path_len - 1;
            return;
        }
        
        // Write to path array with bounds check
        size_t write_index = ant_id * max_steps_per_ant + path_len;
        d_ant_paths[write_index] = current;
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

// CUDA kernel: Deposit pheromones for successful ants and near-solutions
__global__ void deposit_pheromones_kernel(
    DeviceState* d_ant_paths,
    int* d_ant_path_lengths,
    int* d_ant_found_goal,
    int* d_ant_best_distances,    // NEW: closest distance to goal achieved
    float* d_pheromones,
    int pheromone_size,
    int num_ants,
    int max_steps,
    float deposit_amount,
) {
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= num_ants) return;
    
    // Reward ants that found the goal (highest priority)
    if (d_ant_found_goal[ant_id] == 1) {
        int path_len = d_ant_path_lengths[ant_id];
        float quality = deposit_amount / (1.0f + path_len);
        
        for (int i = 0; i < path_len; ++i) {
            DeviceState state = d_ant_paths[ant_id * max_steps + i];
            size_t hash_idx = state.hash() % pheromone_size;
            atomicAdd(&d_pheromones[hash_idx], quality);
        }
    }
    else if (d_ant_best_distances[ant_id] >= 0) {
        int path_len = d_ant_path_lengths[ant_id];
        int best_dist = d_ant_best_distances[ant_id];
        
        // Quality inversely proportional to distance from goal
        // Lower distance = better quality
        float quality = deposit_amount / (1.0f + best_dist + path_len * 0.5f);
        
        for (int i = 0; i < path_len; ++i) {
            DeviceState state = d_ant_paths[ant_id * max_steps + i];
            size_t hash_idx = state.hash() % pheromone_size;
            atomicAdd(&d_pheromones[hash_idx], quality * 0.5f);  // Half weight for partial solutions
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
    // Validate inputs
    if (weights.empty()) {
        std::cerr << "ERROR: weights vector is empty!" << std::endl;
        return {};
    }
    
    // Convert to device states
    DeviceState d_start = to_device_state(start);
    DeviceState d_goal = to_device_state(goal);
    
    // Validate weights size matches number of tiles
    if (weights.size() < (size_t)d_start.num_tiles) {
        std::cerr << "ERROR: weights size (" << weights.size() 
                  << ") is less than num_tiles (" << d_start.num_tiles << ")" << std::endl;
        return {};
    }
    
    // Allocate device memory
    int pheromone_table_size = 100000; // Hash table size
    float* d_pheromones;
    int* d_weights;
    DeviceState* d_ant_paths;
    int* d_ant_path_lengths;
    int* d_ant_found_goal;
    int* d_ant_best_distances;
    curandState* d_rand_states;
    
    CHECK_CUDA(cudaMalloc(&d_pheromones, pheromone_table_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, weights.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ant_paths, params.num_ants * params.max_steps_per_ant * sizeof(DeviceState)));
    CHECK_CUDA(cudaMalloc(&d_ant_path_lengths, params.num_ants * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ant_found_goal, params.num_ants * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ant_best_distances, params.num_ants * sizeof(int)));  // NEW
    CHECK_CUDA(cudaMalloc(&d_rand_states, params.num_ants * sizeof(curandState)));    
    // Initialize pheromones
    std::vector<float> init_pheromones(pheromone_table_size, params.initial_pheromone);
    CHECK_CUDA(cudaMemcpy(d_pheromones, init_pheromones.data(), pheromone_table_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize random states (do this once, not every iteration)
    int threads_per_block = 64;  // Reduced for curand_init
    int blocks = (params.num_ants + threads_per_block - 1) / threads_per_block;
    init_curand_kernel<<<blocks, threads_per_block>>>(d_rand_states, params.num_ants, 12345ULL);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Best solution tracking
    std::vector<State> best_solution;
    int best_length = INT_MAX;
    int total_visited = 0;
    
    // ACO main loop
    threads_per_block = 128;  // Can use more now that curand_init is separate
    blocks = (params.num_ants + threads_per_block - 1) / threads_per_block;
    
    for (int iter = 0; iter < params.max_iterations; ++iter) {
        // Construct solutions
        aco_construct_solutions_kernel<<<blocks, threads_per_block>>>(
            d_start, d_goal, d_weights, d_pheromones, pheromone_table_size,
            params.num_ants, params.max_steps_per_ant, params.alpha, params.beta,
            d_ant_paths, d_ant_path_lengths, d_ant_found_goal, d_ant_best_distances,
            d_rand_states
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy results back
        std::vector<int> h_path_lengths(params.num_ants);
        std::vector<int> h_found_goal(params.num_ants);
        std::vector<int> h_best_distances(params.num_ants);
        cudaMemcpy(h_path_lengths.data(), d_ant_path_lengths, params.num_ants * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_found_goal.data(), d_ant_found_goal, params.num_ants * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_distances.data(), d_ant_best_distances, params.num_ants * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Count solutions found this iteration
        int solutions_found = 0;
        for (int ant = 0; ant < params.num_ants; ++ant) {
            if (h_found_goal[ant] == 1) solutions_found++;
        }
        
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
            d_ant_paths, d_ant_path_lengths, d_ant_found_goal, d_ant_best_distances,
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
    cudaFree(d_ant_best_distances);  // NEW
    cudaFree(d_rand_states);
    
    if (visited_nodes) *visited_nodes = total_visited;
    
    return best_solution;
}
