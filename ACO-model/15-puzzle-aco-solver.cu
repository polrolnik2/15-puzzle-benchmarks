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
#include <chrono>

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
    float exploitation_prob,       // ACS q0: greedy pick probability
    float local_evaporation,       // ACS local decay after each move
    float initial_pheromone,       // tau0 for local update
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
    
    // Tabu list: track visited state hashes to prevent cycles
    // Using a simple hash set with max_steps_per_ant capacity
    size_t tabu_list[1000];  // Reasonable max for 15-puzzle
    int tabu_size = 0;
    
    // Track the best (minimum) distance to goal this ant achieves
    int best_distance = manhattan_distance_device(start_state, goal_state, d_weights);
    d_ant_best_distances[ant_id] = best_distance;
    
    d_ant_paths[ant_id * max_steps_per_ant] = current;
    d_ant_found_goal[ant_id] = 0;
    
    // Add start state to tabu list
    if (tabu_size < 1000) {
        tabu_list[tabu_size++] = current.hash();
    }
    
    for (int step = 0; step < max_steps_per_ant; ++step) {
        // Check if goal reached
        if (current == goal_state) {
            d_ant_found_goal[ant_id] = 1;
            d_ant_path_lengths[ant_id] = path_len;
            rand_states[ant_id] = local_rand_state;
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
            rand_states[ant_id] = local_rand_state;
            return;
        }
        
        // Filter out moves that lead to states in tabu list (already visited)
        DeviceState valid_moves[4];
        int valid_count = 0;
        for (int i = 0; i < num_moves; ++i) {
            size_t move_hash = moves[i].hash();
            bool in_tabu = false;
            for (int t = 0; t < tabu_size; ++t) {
                if (tabu_list[t] == move_hash) {
                    in_tabu = true;
                    break;
                }
            }
            if (!in_tabu) {
                valid_moves[valid_count++] = moves[i];
            }
        }
        
        // If all moves are tabu, we're stuck in a dead end
        if (valid_count == 0) {
            d_ant_found_goal[ant_id] = -7;  // Dead end code
            d_ant_path_lengths[ant_id] = path_len;
            rand_states[ant_id] = local_rand_state;
            return;
        }
        
        // Calculate desirability tau^alpha * eta^beta for valid moves only
        float desirability[4];
        float total_prob = 0.0f;
        int best_idx = 0;
        float best_score = -1.0f;

        for (int i = 0; i < valid_count; ++i) {
            size_t hash_idx = valid_moves[i].hash() % pheromone_size;
            float pheromone = fmaxf(d_pheromones[hash_idx], 0.01f); 
            float heuristic = 1.0f / (1.0f + manhattan_distance_device(valid_moves[i], goal_state, d_weights));
            assert(heuristic > 0.0f && heuristic <= 1.0f && "ERROR: Invalid heuristic value!");
            float score = powf(pheromone, alpha) * powf(heuristic, beta);
            assert(score >= 0.0f && score < 1e9f && "ERROR: Invalid desirability!");
            desirability[i] = score;
            total_prob += score;
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
        assert(total_prob > 0.0f && total_prob < 1e9f && "ERROR: Invalid total probability!");
        int selected = 0;
        float q = curand_uniform(&local_rand_state);
        if (q <= exploitation_prob) {
            selected = best_idx;
        } else {
            float rand_val = curand_uniform(&local_rand_state) * total_prob;
            float cumulative = 0.0f;
            for (int i = 0; i < valid_count; ++i) {
                cumulative += desirability[i];
                if (rand_val <= cumulative) {
                    selected = i;
                    break;
                }
            }
        }
        assert(selected >= 0 && selected < valid_count && "ERROR: Invalid move selection!");
        current = valid_moves[selected];
        path_len++;
        
        // Add new state to tabu list
        if (tabu_size < 500) {
            tabu_list[tabu_size++] = current.hash();
        }
        
        if (path_len >= max_steps_per_ant) {
            d_ant_found_goal[ant_id] = -6;
            d_ant_path_lengths[ant_id] = path_len;
            rand_states[ant_id] = local_rand_state;
            return;
        }
        
        size_t write_index = ant_id * max_steps_per_ant + path_len;
        d_ant_paths[write_index] = current;

        // ACS local pheromone update on chosen move (state hash)
        size_t chosen_hash = current.hash() % pheromone_size;
        float tau_old = d_pheromones[chosen_hash];
        float tau_new = (1.0f - local_evaporation) * tau_old + local_evaporation * initial_pheromone;
        atomicExch(&d_pheromones[chosen_hash], fmaxf(tau_new, 0.0f));
        
        // Update best distance if this state is closer to goal
        int current_distance = manhattan_distance_device(current, goal_state, d_weights);
        best_distance = current_distance;
    }
    
    d_ant_best_distances[ant_id] = best_distance;
    d_ant_path_lengths[ant_id] = path_len;
    rand_states[ant_id] = local_rand_state;
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
    int* d_ant_best_distances,
    float* d_pheromones,
    int pheromone_size,
    int best_ant_id,            // ID of ant with best solution
    int best_path_length,       // length of best path
    int max_steps_per_ant,      // buffer size per ant
    float deposit_amount
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= best_path_length) return;
    DeviceState state = d_ant_paths[best_ant_id * max_steps_per_ant + idx];
    size_t hash_idx = state.hash() % pheromone_size;
    float quality = deposit_amount;
    atomicAdd(&d_pheromones[hash_idx], quality);
}

/* 
* @brief Host function to solve 15-puzzle using ACO on GPU.
* @param start The starting state of the puzzle.
* @param goal The goal state of the puzzle.
* @param weights Weights for each tile (size should match number of tiles).
* @param params ACO parameters.
* @param visited_nodes Pointer to store number of visited nodes (can be nullptr).
* @return Vector of States representing the best solution path found.
*/
std::vector<State> PuzzleSolveACO(
    const State &start, 
    const State &goal, 
    std::vector<int> weights,
    const ACOParams& params,
    int* visited_nodes,
    int* out_distance
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
    CHECK_CUDA(cudaMalloc(&d_ant_best_distances, params.num_ants * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rand_states, params.num_ants * sizeof(curandState)));    
    // Initialize pheromones
    std::vector<float> init_pheromones(pheromone_table_size, params.initial_pheromone);
    CHECK_CUDA(cudaMemcpy(d_pheromones, init_pheromones.data(), pheromone_table_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    int threads_per_block = 64;
    int blocks = (params.num_ants + threads_per_block - 1) / threads_per_block;
    unsigned long long seed = std::chrono::steady_clock::now().time_since_epoch().count();
    init_curand_kernel<<<blocks, threads_per_block>>>(d_rand_states, params.num_ants, seed);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<State> best_solution;
    int best_length = INT_MAX;
    int total_visited = 0;
    
    threads_per_block = 128;  
    blocks = (params.num_ants + threads_per_block - 1) / threads_per_block;

    int overall_best_ant_distance = INT_MAX;
    
    for (int iter = 0; iter < params.max_iterations; ++iter) {
        aco_construct_solutions_kernel<<<blocks, threads_per_block>>>(
            d_start, d_goal, d_weights, d_pheromones, pheromone_table_size,
            params.num_ants, params.max_steps_per_ant, params.alpha, params.beta,
            params.exploitation_prob, params.local_evaporation, params.initial_pheromone,
            d_ant_paths, d_ant_path_lengths, d_ant_found_goal, d_ant_best_distances,
            d_rand_states
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<int> h_path_lengths(params.num_ants);
        std::vector<int> h_found_goal(params.num_ants);
        std::vector<int> h_best_distances(params.num_ants);
        cudaMemcpy(h_path_lengths.data(), d_ant_path_lengths, params.num_ants * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_found_goal.data(), d_ant_found_goal, params.num_ants * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_distances.data(), d_ant_best_distances, params.num_ants * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Determine iteration-best ant (prefer complete solutions, else closest)
        int best_ant = -1;
        int best_ant_distance = INT_MAX;
        
        // Totals and best-so-far update
        for (int ant = 0; ant < params.num_ants; ++ant) {
            total_visited += h_path_lengths[ant];
            if (h_found_goal[ant] == 1) {
                if (h_path_lengths[ant] < best_length) {
                    best_length = h_path_lengths[ant];
                    best_ant = ant;
                    best_ant_distance = 0;
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
            } else {
                if (h_best_distances[ant] < best_ant_distance) {
                    best_length = h_path_lengths[ant];
                    best_ant_distance = h_best_distances[ant];
                    best_ant = ant;
                }
            }
            overall_best_ant_distance = best_ant_distance;
        }
        
        // Evaporate pheromones
        int evap_blocks = (pheromone_table_size + threads_per_block - 1) / threads_per_block;
        evaporate_pheromones_kernel<<<evap_blocks, threads_per_block>>>(
            d_pheromones, pheromone_table_size, params.evaporation_rate, 0.01f
        );
        
        // Deposit pheromones on complete solution (iteration-best)
        if (best_ant >= 0 && h_found_goal[best_ant] == 1 && h_path_lengths[best_ant] > 0) {
            int best_path_len = h_path_lengths[best_ant];
            int deposit_blocks = (best_path_len + threads_per_block - 1) / threads_per_block;
            deposit_pheromones_kernel<<<deposit_blocks, threads_per_block>>>(
                d_ant_paths, d_ant_path_lengths, d_ant_found_goal, d_ant_best_distances,
                d_pheromones, pheromone_table_size, best_ant, best_path_len,
                params.max_steps_per_ant, params.pheromone_deposit
            );
            CHECK_CUDA(cudaGetLastError());
        }
        else if (best_ant >= 0 && h_path_lengths[best_ant] > 0 && best_ant_distance < INT_MAX) {
            int best_path_len = h_path_lengths[best_ant];
            int deposit_blocks = (best_path_len + threads_per_block - 1) / threads_per_block;
            float scaled_deposit = params.pheromone_deposit;
            deposit_pheromones_kernel<<<deposit_blocks, threads_per_block>>>(
                d_ant_paths, d_ant_path_lengths, d_ant_found_goal, d_ant_best_distances,
                d_pheromones, pheromone_table_size, best_ant, best_path_len,
                params.max_steps_per_ant, scaled_deposit
            );
            CHECK_CUDA(cudaGetLastError());
        }
        cudaDeviceSynchronize();
    }
    
    // Cleanup
    cudaFree(d_pheromones);
    cudaFree(d_weights);
    cudaFree(d_ant_paths);
    cudaFree(d_ant_path_lengths);
    cudaFree(d_ant_found_goal);
    cudaFree(d_ant_best_distances);
    cudaFree(d_rand_states);
    
    if (visited_nodes) *visited_nodes = total_visited;
    if (out_distance) *out_distance = overall_best_ant_distance;    
    
    return best_solution;
}
