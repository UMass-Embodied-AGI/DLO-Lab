import argparse
import subprocess
import os
import time
import json
from datetime import datetime


def check_completion(args):
    record_path = f"logs/{args.task}/{args.exp_name}/ckpts/record.json"
    if not os.path.exists(record_path):
        return False
    with open(record_path, 'r') as f:
        record = json.load(f)
    epoch = int(record.get('epoch', 0)) + 1
    return epoch >= args.n_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wiring_ring', help='Task name')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_envs', type=int, default=100)
    parser.add_argument('--n_traj', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_substeps_per_step', type=int, default=20)
    parser.add_argument('--steps_interval_split', type=int, default=1)
    # parser.add_argument('--per_fit_ratio', type=int, default=1)
    parser.add_argument('--bound', type=float, default=0.01)
    parser.add_argument('--angle_bound', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument("--delay", type=float, default=2.0,
                    help="Seconds to wait before restarting after exit (default: 2.0)")
    args = parser.parse_args()

    script_file = None
    if 'rudin' in args.exp_name:
        script_file = 'rl/rudinppo.py'
    elif 'sac' in args.exp_name:
        script_file = 'rl/sac.py'

    if script_file is None:
        print("[supervisor] Error: Cannot determine RL algorithm from exp_name.")
        exit(0)

    cmd = [
        "python", script_file,
        "--task", args.task,
        "--exp_name", args.exp_name,
        "--n_epochs", str(args.n_epochs),
        "--n_envs", str(args.n_envs),
        "--n_traj", str(args.n_traj),
        "--n_steps", str(args.n_steps),
        "--n_substeps_per_step", str(args.n_substeps_per_step),
        "--steps_interval_split", str(args.steps_interval_split),
        "--bound", str(args.bound),
        "--angle_bound", str(args.angle_bound),
        "--seed", str(args.seed)
    ]

    print(f"[supervisor] Starting loop. Will run: {' '.join(cmd)}")
    try:
        i = 1
        while True:
            if check_completion(args):
                print(f"[supervisor] Detected completion for task '{args.task}' "
                      f"with exp_name '{args.exp_name}', max_iter '{args.n_epochs}'. Exiting.")
                break
            print(f"\n[supervisor] Launch #{i} at {datetime.now().isoformat(timespec='seconds')}")
            # Run the child; do not raise on non-zero (we want to restart regardless)
            result = subprocess.run(cmd)
            print(f"[supervisor] Child exited with return code {result.returncode}")
            i += 1
            if args.delay > 0:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\n[supervisor] Stopped by user. Bye!")

if __name__ == "__main__":
    main()
