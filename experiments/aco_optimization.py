import optuna
import ctypes
import os
import tempfile
from random import random
import csv
import matplotlib.pyplot as plt
import argparse

# @Experiment: aco_optimization.py
# @Description: An experiment aiming to find the parameters at which the Ant Colony Optimization algorithm performs best on the 15-puzzle problem.

class ACOArgs:
    def __init__(self, num_ants: int, max_steps: int, alpha: float, beta: float,
                 evaporation: float, deposit: float, initial_pheromone: float,
                 q0: float, local_evaporation: float):
        self.num_ants = num_ants
        self.max_steps = max_steps
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.deposit = deposit
        self.initial_pheromone = initial_pheromone
        self.q0 = q0
        self.local_evaporation = local_evaporation

def load_lib(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Library not found: {path}")
    return ctypes.CDLL(path)


def run_datagen_random_walk(lib, side: int, empty: int, depth: int, seed: int, out_dir: str) -> str:
    fn = lib.datagen_random_walk_to_file
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
                   ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    buf_len = 1024
    out_buf = ctypes.create_string_buffer(buf_len)
    rc = fn(side, empty, depth, ctypes.c_uint(seed), out_dir.encode('utf-8'), out_buf, buf_len)
    if rc != 0:
        raise RuntimeError(f"datagen_random_walk_to_file failed with code {rc}")
    return out_buf.value.decode('utf-8')


def run_aco(lib, instance_path: str, side: int, empty: int, iterations: int, params, weights):
    aco_fn = lib.aco_run_instance
    aco_fn.argtypes = [
        ctypes.c_char_p,  # input_file
        ctypes.c_int,     # side_size
        ctypes.c_int,     # empty_cells
        ctypes.c_int,     # max_iterations
        ctypes.c_int,     # num_ants
        ctypes.c_int,     # max_steps_per_ant
        ctypes.c_float,   # alpha
        ctypes.c_float,   # beta
        ctypes.c_float,   # evaporation_rate
        ctypes.c_float,   # pheromone_deposit
        ctypes.c_float,   # initial_pheromone
        ctypes.c_float,   # exploitation_prob (q0)
        ctypes.c_float,   # local_evaporation
        ctypes.POINTER(ctypes.c_int),  # weights
        ctypes.c_int,     # weights_len
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    aco_fn.restype = ctypes.c_int

    w_arr = (ctypes.c_int * len(weights))(*weights)
    t_ms = ctypes.c_double(0.0)
    steps = ctypes.c_int(0)
    visited = ctypes.c_int(0)
    distance = ctypes.c_int(0)

    rc = aco_fn(
        instance_path.encode('utf-8'),
        side,
        empty,
        iterations,
        params.num_ants,
        params.max_steps,
        ctypes.c_float(params.alpha),
        ctypes.c_float(params.beta),
        ctypes.c_float(params.evaporation),
        ctypes.c_float(params.deposit),
        ctypes.c_float(params.initial_pheromone),
        ctypes.c_float(params.q0),
        ctypes.c_float(params.local_evaporation),
        w_arr,
        len(weights),
        ctypes.byref(t_ms),
        ctypes.byref(steps),
        ctypes.byref(visited),
        ctypes.byref(distance)
    )
    return rc, t_ms.value, steps.value, visited.value, distance.value

def objective(trial):
    # 1. Suggest parameters
    alpha = 1.0
    beta = trial.suggest_float("beta", 0.5, 20.0)
    q0 = trial.suggest_float("q0", 0.7, 0.99)
    evaporation = trial.suggest_float("evaporation", 0.01, 0.5)
    local_evaporation = trial.suggest_float("local_evaporation", 0.01, 0.5)
    initial_pheromone = trial.suggest_float("initial_pheromone", 0.01, 1.0)
    deposit = trial.suggest_float("deposit", 1.0, 6.0)
    num_ants = trial.suggest_int("num_ants", 8, 256)
    max_steps = trial.suggest_int("max_steps", 10, 1000)

    out_dir = tempfile.mkdtemp(prefix="puzzle_aco_datagen_")
    os.makedirs(out_dir, exist_ok=True)

    params = ACOArgs(
        num_ants=num_ants,
        max_steps=max_steps,
        alpha=alpha,
        beta=beta,
        evaporation=evaporation,
        deposit=deposit,
        initial_pheromone=initial_pheromone,
        q0=q0,
        local_evaporation=local_evaporation,
    )

    seed = int(random() * 10000)
    
    # 2. Run on a representative set of puzzles
    scores = []
    
    for i in range(200):
        inst_path = run_datagen_random_walk(datagen_lib, 4, 1, 50, seed, out_dir)
        # Run your ACS with the suggested params
        rc, t_ms, steps, visited, distance = run_aco(
            aco_lib,
            inst_path,
            4,
            1,
            10,
            params,
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        )
        
        # Use a "Distance from Goal" metric if the solution isn't found
        scores.append(distance + steps)
    
    # 3. Return the average (this is what Optuna minimizes)
    return sum(scores) / 200

ap = argparse.ArgumentParser(description="Benchmark CUDA ACO solver against A* baseline.")
ap.add_argument('--datagen-lib', required=True, help='Path to datagen shared library')
ap.add_argument('--aco-lib', required=True, help='Path to acomodel shared library')
args = ap.parse_args()

datagen_lib = load_lib(args.datagen_lib)
aco_lib = load_lib(args.aco_lib)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)