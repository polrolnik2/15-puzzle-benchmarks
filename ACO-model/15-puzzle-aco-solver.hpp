#ifndef PUZZLE_ACO_SOLVER_HPP
#define PUZZLE_ACO_SOLVER_HPP

#include <vector>
#include "state.hpp"

/**
 * @file 15-puzzle-aco-solver.hpp
 * @brief CUDA-accelerated Ant Colony Optimization solver for N-puzzle.
 */

/**
 * @brief ACO algorithm parameters.
 * @var num_ants Number of ants per iteration.
 * @var max_iterations Maximum number of ACO iterations.
 * @var max_steps_per_ant Maximum steps each ant can take.
 * @var alpha Pheromone importance factor.
 * @var beta Heuristic importance factor.
 * @var evaporation_rate Global pheromone evaporation rate (0-1).
 * @var pheromone_deposit Amount of pheromone deposited by successful ants.
 * @var initial_pheromone Initial pheromone level on all paths.
 * @var exploitation_prob ACS q0: probability of greedy selection vs roulette.
 * @var local_evaporation Local ACS pheromone decay applied after each move.
 */
struct ACOParams {
    int num_ants;              // Number of ants per iteration
    int max_iterations;        // Maximum ACO iterations
    int max_steps_per_ant;     // Maximum steps each ant can take
    float alpha;               // Pheromone importance
    float beta;                // Heuristic importance
    float evaporation_rate;    // Global pheromone evaporation (0-1)
    float pheromone_deposit;   // Amount deposited by successful ants
    float initial_pheromone;   // Initial pheromone level
    float exploitation_prob;   // q0: probability of greedy choice
    float local_evaporation;   // Local pheromone decay after each move
    
    ACOParams() : 
        num_ants(256),
        max_iterations(100),
        max_steps_per_ant(100),
        alpha(1.0f),
        beta(2.0f),
        evaporation_rate(0.1f),
        pheromone_deposit(1.0f),
        initial_pheromone(0.1f),
        exploitation_prob(0.9f),
        local_evaporation(0.1f) {}
};

/**
 * @brief Solve the puzzle using CUDA-accelerated ACO.
 *
 * @param start Starting puzzle state.
 * @param goal Goal puzzle state.
 * @param weights Per-tile weights used by the heuristic.
 * @param params ACO algorithm parameters.
 * @param visited_nodes Optional out-parameter to receive number of visited nodes.
 * @return Sequence of states from start to goal (empty if no solution found).
 */
std::vector<State> PuzzleSolveACO(
    const State &start, 
    const State &goal, 
    std::vector<int> weights,
    const ACOParams& params = ACOParams(),
    int* visited_nodes = nullptr,
    int * out_distance = nullptr
);

#endif // PUZZLE_ACO_SOLVER_HPP
