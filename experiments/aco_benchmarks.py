import argparse
import ctypes
import os
import tempfile
from random import random
import csv
import matplotlib.pyplot as plt


# Lightweight ctypes helpers -------------------------------------------------

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


def run_astar(lib, instance_path: str, side: int, empty: int, weights):
    """Run A* on an instance and return (rc, time_ms, steps, visited)."""
    astar_fn = lib.astar_run_instance
    astar_fn.argtypes = [
        ctypes.c_char_p,  # input_file
        ctypes.c_int,     # side_size
        ctypes.c_int,     # empty_cells
        ctypes.POINTER(ctypes.c_int),  # weights
        ctypes.c_int,     # weights_len
        ctypes.POINTER(ctypes.c_double),  # out_time_ms
        ctypes.POINTER(ctypes.c_int),  # out_steps
        ctypes.POINTER(ctypes.c_int),  # out_visited
    ]
    astar_fn.restype = ctypes.c_int

    w_arr = (ctypes.c_int * len(weights))(*weights)
    t_ms = ctypes.c_double(0.0)
    steps = ctypes.c_int(0)
    visited = ctypes.c_int(0)

    rc = astar_fn(
        instance_path.encode('utf-8'),
        side,
        empty,
        w_arr,
        len(weights),
        ctypes.byref(t_ms),
        ctypes.byref(steps),
        ctypes.byref(visited),
    )
    return rc, t_ms.value, steps.value, visited.value


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


# Benchmark driver -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark CUDA ACO solver against A* baseline.")
    ap.add_argument('--datagen-lib', required=True, help='Path to datagen shared library')
    ap.add_argument('--aco-lib', required=True, help='Path to acomodel shared library')
    ap.add_argument('--astar-lib', required=True, help='Path to astarmodel shared library (for baseline)')
    ap.add_argument('--iterations', required=True, help='Comma-separated iteration counts (e.g., 25,50,100)')
    ap.add_argument('--depths', required=True, help='Comma-separated depths to generate')
    ap.add_argument('--samples-per-depth', type=int, default=3, help='Samples per depth to average')
    ap.add_argument('--seed', type=int, default=int(random()*1e6), help='Base RNG seed')
    ap.add_argument('--tmp-dir', type=str, default=None, help='Directory to store generated instances')
    ap.add_argument('--side', type=int, default=4, help='Puzzle side length')
    ap.add_argument('--empty', type=int, default=1, help='Number of empty cells')
    ap.add_argument('--weights', type=str, default='1,1,1,1,1,1,1,1,1,1,1,1,1,1,1', help='Comma-separated tile weights')
    ap.add_argument('--num-ants', type=int, default=256, help='Ants per iteration')
    ap.add_argument('--max-steps', type=int, default=100, help='Max steps per ant')
    ap.add_argument('--alpha', type=float, default=1.0, help='Pheromone importance')
    ap.add_argument('--beta', type=float, default=2.0, help='Heuristic importance')
    ap.add_argument('--evaporation', type=float, default=0.1, help='Evaporation rate (0-1)')
    ap.add_argument('--deposit', type=float, default=1.0, help='Pheromone deposit amount')
    ap.add_argument('--initial-pheromone', type=float, default=0.1, help='Initial pheromone level')
    ap.add_argument('--q0', type=float, default=0.9, help='ACS exploitation probability (0-1)')
    ap.add_argument('--local-evaporation', type=float, default=0.1, help='ACS local pheromone decay')
    ap.add_argument('--log-csv', type=str, help='Optional CSV path for logging results')
    ap.add_argument('--plot-file', type=str, help='Path to save plot (e.g., aco_vs_astar.png)')
    args = ap.parse_args()

    datagen_lib = load_lib(args.datagen_lib)
    aco_lib = load_lib(args.aco_lib)
    astar_lib = load_lib(args.astar_lib)

    iteration_list = [int(x) for x in args.iterations.split(',') if x.strip()]
    depth_list = [int(x) for x in args.depths.split(',') if x.strip()]
    weights = [int(x) for x in args.weights.split(',') if x.strip()]

    params = ACOArgs(
        num_ants=args.num_ants,
        max_steps=args.max_steps,
        alpha=args.alpha,
        beta=args.beta,
        evaporation=args.evaporation,
        deposit=args.deposit,
        initial_pheromone=args.initial_pheromone,
        q0=args.q0,
        local_evaporation=args.local_evaporation,
    )

    out_dir = args.tmp_dir or tempfile.mkdtemp(prefix="puzzle_aco_datagen_")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    base_seed = int(args.seed)

    for depth_idx, depth in enumerate(sorted(depth_list)):
        print(f"Depth {depth}: generating {args.samples_per_depth} sample(s)...")
        for sample in range(args.samples_per_depth):
            seed = base_seed + depth_idx * 1000 + sample
            inst_path = run_datagen_random_walk(datagen_lib, args.side, args.empty, depth, seed, out_dir)

            # Run A* baseline once per instance
            rc_astar, t_astar, steps_astar, visited_astar = run_astar(
                astar_lib,
                inst_path,
                args.side,
                args.empty,
                weights,
            )
            astar_optimal = steps_astar if rc_astar > 0 else None
            print(f"  A* baseline: {steps_astar} steps in {t_astar:.2f} ms" if rc_astar > 0 else "  A* baseline: FAILED")

            for iters in iteration_list:
                rc, t_ms, steps, visited, distance = run_aco(
                    aco_lib,
                    inst_path,
                    args.side,
                    args.empty,
                    iters,
                    params,
                    weights,
                )
                quality_ratio = (steps / astar_optimal) if (astar_optimal and rc > 0) else None
                results.append({
                    'param_depth': depth,
                    'instance': inst_path,
                    'iterations': iters,
                    'astar_rc': rc_astar,
                    'astar_steps': steps_astar,
                    'aco_rc': rc,
                    'aco_steps': steps,
                    'aco_time_ms': t_ms,
                    'aco_visited': visited,
                    'quality_ratio': quality_ratio,
                })
                status = "OK" if rc > 0 else "NO-SOLUTION"
                qual_str = f" ({quality_ratio:.2f}x A*)" if quality_ratio else ""
                print(f"  iter={iters:4d} sample={sample+1}/{args.samples_per_depth}: {status}, steps={steps}{qual_str}, distance={distance}, time={t_ms:.2f} ms")

    if args.log_csv:
        with open(args.log_csv, 'w', newline='') as f:
            fieldnames = ['param_depth', 'instance', 'iterations', 'astar_rc', 'astar_steps', 'aco_rc', 'aco_steps', 'aco_time_ms', 'aco_visited', 'quality_ratio']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results logged to {args.log_csv}")

    # Plot only successful ACO runs with A* baselines (rc > 0 and quality_ratio available)
    successes = [r for r in results if r['aco_rc'] > 0 and r['quality_ratio'] is not None]
    if not successes:
        print("No successful ACO runs with A* baseline to plot.")
        return

    iter_vals = [r['iterations'] for r in successes]
    quality_vals = [r['quality_ratio'] for r in successes]
    time_vals = [r['aco_time_ms'] for r in successes]
    depth_vals = [r['param_depth'] for r in successes]

    fig, (ax_quality, ax_time) = plt.subplots(1, 2, figsize=(12, 5))

    scatter = ax_quality.scatter(iter_vals, quality_vals, c=depth_vals, cmap='viridis', alpha=0.75, s=60)
    ax_quality.set_xlabel('ACO Iterations')
    ax_quality.set_ylabel('Quality Ratio (ACO Steps / A* Steps)')
    ax_quality.set_title('ACO Quality vs A* Baseline')
    ax_quality.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='A* optimal')
    ax_quality.grid(True, alpha=0.3)
    ax_quality.legend()

    ax_time.scatter(iter_vals, time_vals, c=depth_vals, cmap='viridis', alpha=0.75, s=60)
    ax_time.set_xlabel('ACO Iterations')
    ax_time.set_ylabel('Time (ms)')
    ax_time.set_title('Runtime vs Iterations')
    ax_time.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=[ax_quality, ax_time], label='Generation Depth')

    fig.tight_layout()

    if args.plot_file:
        fig.savefig(args.plot_file, dpi=150)
        print(f"Plot saved to {args.plot_file}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
