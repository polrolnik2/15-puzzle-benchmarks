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
 * @var evaporation_rate Pheromone evaporation rate (0-1).
 * @var pheromone_deposit Amount of pheromone deposited by successful ants.
 * @var initial_pheromone Initial pheromone level on all paths.
 */
struct ACOParams {
    int num_ants;              // Number of ants per iteration
    int max_iterations;        // Maximum ACO iterations
    int max_steps_per_ant;     // Maximum steps each ant can take
    float alpha;               // Pheromone importance
    float beta;                // Heuristic importance
    float evaporation_rate;    // Pheromone evaporation (0-1)
    float pheromone_deposit;   // Amount deposited by successful ants
    float initial_pheromone;   // Initial pheromone level
    
    ACOParams() : 
        num_ants(256),
        max_iterations(100),
        max_steps_per_ant(100),
        alpha(1.0f),
        beta(2.0f),
        evaporation_rate(0.1f),
        pheromone_deposit(1.0f),
        initial_pheromone(0.1f) {}
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
    int* visited_nodes = nullptr
);

#endif // PUZZLE_ACO_SOLVER_HPP
