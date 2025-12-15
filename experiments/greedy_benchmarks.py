import argparse
import csv
import ctypes
import os
import tempfile
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


def run_weighted_astar(lib, instance_path, side, empty, weights, heuristic_weight):
    wa_fn = lib.weighted_astar_run_instance
    wa_fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                      ctypes.c_float,
                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    wa_fn.restype = ctypes.c_int
    w_arr = (ctypes.c_int * len(weights))(*weights)
    t = ctypes.c_double(0.0)
    steps = ctypes.c_int(0)
    visited = ctypes.c_int(0)
    rc = wa_fn(instance_path.encode('utf-8'), side, empty,
               w_arr, len(weights), ctypes.c_float(heuristic_weight),
               ctypes.byref(t), ctypes.byref(steps), ctypes.byref(visited))
    return rc, t.value, steps.value, visited.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datagen-lib', required=True, help='Path to datagen shared library (enables on-the-fly generation)')
    ap.add_argument('--depths', type=str, required=True, help='Comma-separated depths to sample (e.g., 5,10,15)')
    ap.add_argument('--samples-per-depth', type=int, default=5, help='Samples per depth to average')
    ap.add_argument('--seed', type=int, default=12345, help='Base RNG seed')
    ap.add_argument('--tmp-dir', type=str, default=None, help='Directory to store generated instances (defaults to temp)')
    ap.add_argument('--side', type=int, default=4)
    ap.add_argument('--empty', type=int, default=1)
    ap.add_argument('--weights', type=str, default='1,1,1,1,1,1,1,1,1,1,1,1,1,1,1', help='Comma-separated tile weights')
    ap.add_argument('--heuristic-weights', type=str, default='1.0,1.5,2.0', help='Comma-separated heuristic weights for weighted A*')
    ap.add_argument('--bfs-lib', required=True, help='Path to bfsmodel shared library')
    ap.add_argument('--wastar-lib', required=True, help='Path to weighted A* shared library')
    ap.add_argument('--log-csv', type=str, help='Path to output CSV log file')
    ap.add_argument('--plot-file', type=str, help='Path to save runtime plot (accuracy plot gets _accuracy suffix)')
    args = ap.parse_args()

    bfs_lib = load_lib(args.bfs_lib)
    wastar_lib = load_lib(args.wastar_lib)
    datagen_lib = load_lib(args.datagen_lib)

    weights = [int(x) for x in args.weights.split(',') if x.strip()]
    heuristic_weights = [float(x) for x in args.heuristic_weights.split(',') if x.strip()]

    results = []

    depth_list = [int(x) for x in args.depths.split(',') if x.strip()]
    out_dir = args.tmp_dir or tempfile.mkdtemp(prefix="puzzle_datagen_")
    base_seed = int(args.seed)
    for idx, depth in enumerate(sorted(depth_list)):
        print(f"Processing depth parameter {depth}...")
        for k in range(args.samples_per_depth):
            seed = base_seed + idx * 1000 + k
            inst_path = run_datagen_random_walk(datagen_lib, args.side, args.empty, depth, seed, out_dir)
            rc_b, t_b, steps_b, visited_b = run_bfs(bfs_lib, inst_path, args.side, args.empty)
            if rc_b < 0:
                continue
            for hw in heuristic_weights:
                rc_w, t_w, steps_w, visited_w = run_weighted_astar(wastar_lib, inst_path, args.side, args.empty, weights, hw)
                if rc_w < 0:
                    continue
                results.append({
                    'param_depth': depth,
                    'instance': inst_path,
                    'heuristic_weight': hw,
                    'bfs_steps': steps_b,
                    'bfs_time_ms': t_b,
                    'bfs_visited': visited_b,
                    'wastar_steps': steps_w,
                    'wastar_time_ms': t_w,
                    'wastar_visited': visited_w
                })

    if not results:
        print("No results collected; exiting.")
        return

    if args.log_csv:
        with open(args.log_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'param_depth', 'instance', 'heuristic_weight',
                'bfs_steps', 'bfs_time_ms', 'bfs_visited',
                'wastar_steps', 'wastar_time_ms', 'wastar_visited'
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results logged to {args.log_csv}")

    # Runtime plot: x-axis optimal solution length, colored by heuristic weight
    fig_rt, ax_rt = plt.subplots(figsize=(8, 6))
    unique_hw = sorted(set(r['heuristic_weight'] for r in results))
    for hw in unique_hw:
        xs = [r['bfs_steps'] for r in results if r['heuristic_weight'] == hw]
        ys = [r['wastar_time_ms'] for r in results if r['heuristic_weight'] == hw]
        ax_rt.scatter(xs, ys, label=f'w={hw:g}', alpha=0.7, s=50)
    ax_rt.set_xlabel('Optimal Solution Length (BFS steps)')
    ax_rt.set_ylabel('Weighted A* Time (ms)')
    ax_rt.set_title('Weighted A* Runtime by Solution Length')
    ax_rt.legend()
    ax_rt.grid(True, alpha=0.3)

    # Accuracy plot: x-axis = optimal solution length, grouped by heuristic weight (w > 1)
    fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
    unique_hw_acc = [hw for hw in sorted(set(r['heuristic_weight'] for r in results)) if hw > 1.0]
    for hw in unique_hw_acc:
        xs = [r['bfs_steps'] for r in results if r['heuristic_weight'] == hw and r['wastar_steps'] > 0 and r['bfs_steps'] > 0]
        ys = [r['wastar_steps'] / r['bfs_steps'] for r in results
              if r['heuristic_weight'] == hw and r['wastar_steps'] > 0 and r['bfs_steps'] > 0]
        if not xs:
            continue
        ax_acc.scatter(xs, ys, label=f'w={hw:g}', alpha=0.7, s=50)
        # Mark mean accuracy for this weight at its average solution length
        ax_acc.scatter(sum(xs) / len(xs), sum(ys) / len(ys), c='black', marker='x', s=80)
    ax_acc.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax_acc.set_xlabel('Optimal Solution Length (BFS steps)')
    ax_acc.set_ylabel('Accuracy (found_steps / optimal_steps)')
    ax_acc.set_title('Solution Accuracy vs Length (w > 1)')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    if args.plot_file:
        fig_rt.savefig(args.plot_file, dpi=150)
        base, ext = os.path.splitext(args.plot_file)
        acc_path = f"{base}_accuracy{ext}" if ext else f"{args.plot_file}_accuracy"
        fig_acc.savefig(acc_path, dpi=150)
        print(f"Plots saved to {args.plot_file} and {acc_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
