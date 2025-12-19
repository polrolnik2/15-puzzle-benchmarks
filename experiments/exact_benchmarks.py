import ctypes
import os
import time
import argparse
import tempfile
import csv
import matplotlib.pyplot as plt


def load_lib(path):
    return ctypes.CDLL(path)


def run_datagen_random_walk(lib, side, empty, depth, seed, out_dir):
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


def run_bfs(lib, instance_path, side, empty):
    bfs_fn = lib.bfs_run_instance
    bfs_fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    bfs_fn.restype = ctypes.c_int
    t = ctypes.c_double(0.0)
    steps = ctypes.c_int(0)
    visited = ctypes.c_int(0)
    rc = bfs_fn(instance_path.encode('utf-8'), side, empty, ctypes.byref(t), ctypes.byref(steps), ctypes.byref(visited))
    return rc, t.value, steps.value, visited.value


def run_astar(lib, instance_path, side, empty, weights):
    astar_fn = lib.astar_run_instance
    astar_fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                         ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                         ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    astar_fn.restype = ctypes.c_int
    w_arr = (ctypes.c_int * len(weights))(*weights)
    t = ctypes.c_double(0.0)
    steps = ctypes.c_int(0)
    visited = ctypes.c_int(0)
    rc = astar_fn(instance_path.encode('utf-8'), side, empty,
                  w_arr, len(weights), ctypes.byref(t), ctypes.byref(steps), ctypes.byref(visited))
    return rc, t.value, steps.value, visited.value


def collect_instances(base_dir):
    instances = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.txt') or f.endswith('.state') or f.endswith('.inst'):
                instances.append(os.path.join(root, f))
    instances.sort()
    return instances


# @Experiment: Exact Search Benchmarks
# @Description: Compare BFS and A* search algorithms on generated puzzle instances.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datagen-lib', required=True, help='Path to datagen shared library (enables on-the-fly generation)')
    ap.add_argument('--depths', type=str, help='Comma-separated depths to sample (e.g., 5,10,15)')
    ap.add_argument('--samples-per-depth', type=int, default=5, help='Samples per depth to average')
    ap.add_argument('--seed', type=int, default=12345, help='Base RNG seed')
    ap.add_argument('--tmp-dir', type=str, default=None, help='Directory to store generated instances (defaults to temp)')
    ap.add_argument('--side', type=int, default=4)
    ap.add_argument('--empty', type=int, default=1)
    ap.add_argument('--weights', type=str, default='1,1,1,1,1,1,1,1,1,1,1,1,1,1,1')
    ap.add_argument('--bfs-lib', required=True, help='Path to bfsmodel shared library')
    ap.add_argument('--astar-lib', required=True, help='Path to astarmodel shared library')
    ap.add_argument('--log-csv', type=str, help='Path to output CSV log file')
    ap.add_argument('--plot-file', type=str, help='Path to save plot (e.g., plot.png)')
    args = ap.parse_args()

    bfs_lib = load_lib(args.bfs_lib)
    astar_lib = load_lib(args.astar_lib)

    weights = [int(x) for x in args.weights.split(',') if x.strip()]

    # Store detailed results: each row = (param_depth, actual_steps_bfs, time_bfs, actual_steps_astar, time_astar)
    results = []

    datagen_lib = load_lib(args.datagen_lib)
    # Parse depths
    if not args.depths:
        raise SystemExit("--depths is required when --datagen-lib is provided")
    depth_list = [int(x) for x in args.depths.split(',') if x.strip()]
    # Prepare output directory
    out_dir = args.tmp_dir or tempfile.mkdtemp(prefix="puzzle_datagen_")
    # For reproducibility across depths, vary seed
    base_seed = int(args.seed)
    for idx, depth in enumerate(sorted(depth_list)):
        print(f"Processing depth parameter {depth}...")
        for k in range(args.samples_per_depth):
            seed = base_seed + idx * 1000 + k
            inst_path = run_datagen_random_walk(datagen_lib, args.side, args.empty, depth, seed, out_dir)
            rc_b, t_b, steps_b, visited_b = run_bfs(bfs_lib, inst_path, args.side, args.empty)
            rc_a, t_a, steps_a, visited_a = run_astar(astar_lib, inst_path, args.side, args.empty, weights)
            if rc_b >= 0 and rc_a >= 0:
                # steps are solution length (actual depth); store both algorithms' results
                results.append({
                    'param_depth': depth,
                    'instance': inst_path,
                    'bfs_steps': steps_b,
                    'bfs_time_ms': t_b,
                    'bfs_visited': visited_b,
                    'astar_steps': steps_a,
                    'astar_time_ms': t_a,
                    'astar_visited': visited_a
                })

    # Write CSV log if requested
    if args.log_csv:
        with open(args.log_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['param_depth', 'instance', 'bfs_steps', 'bfs_time_ms', 'bfs_visited',
                                                     'astar_steps', 'astar_time_ms', 'astar_visited'])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results logged to {args.log_csv}")

    # Extract data for plotting: use actual solution length (steps) as x-axis
    bfs_depths = [r['bfs_steps'] for r in results]
    bfs_times = [r['bfs_time_ms'] for r in results]
    bfs_visited = [r['bfs_visited'] for r in results]
    astar_depths = [r['astar_steps'] for r in results]
    astar_times = [r['astar_time_ms'] for r in results]
    astar_visited = [r['astar_visited'] for r in results]
    
    # Calculate effective branching factor: visited / solution_length
    bfs_branching = [v / d if d > 0 else 0 for v, d in zip(bfs_visited, bfs_depths)]
    astar_branching = [v / d if d > 0 else 0 for v, d in zip(astar_visited, astar_depths)]

    # Figure 1: Runtime comparison
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(bfs_depths, bfs_times, c='tab:blue', label='BFS', alpha=0.7, s=50)
    ax1.scatter(astar_depths, astar_times, c='tab:orange', label='A*', alpha=0.7, s=50)
    ax1.set_xlabel('Solution Length ')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Runtime by Solution Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Figure 2: Branching factor comparison
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(bfs_depths, bfs_branching, c='tab:blue', label='BFS', alpha=0.7, s=50)
    ax2.scatter(astar_depths, astar_branching, c='tab:orange', label='A*', alpha=0.7, s=50)
    ax2.set_xlabel('Solution Length (actual steps)')
    ax2.set_ylabel('Effective Branching Factor (visited/steps)')
    ax2.set_title('Branching Factor by Solution Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if args.plot_file:
        # Save runtime plot to provided path
        fig1.savefig(args.plot_file, dpi=150)
        # Derive branching plot filename by inserting _branch before extension
        base, ext = os.path.splitext(args.plot_file)
        branch_path = f"{base}_branch{ext}" if ext else f"{args.plot_file}_branch"
        fig2.savefig(branch_path, dpi=150)
        print(f"Plots saved to {args.plot_file} and {branch_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
