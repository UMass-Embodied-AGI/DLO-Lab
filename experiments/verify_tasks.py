#!/usr/bin/env python3
"""
verify_tasks.py — DLO-Lab installation verification
=====================================================
Builds each of the 8 benchmark environments, runs one reset step,
and reports whether the scene loaded correctly.

Run from the `experiments/` directory:

    cd experiments
    python verify_tasks.py

Requirements: DLO-Lab + Genesis installed, assets downloaded to
`genesis/assets/dlo-lab/` (see INSTALL.md).

Flags
-----
--tasks         space-separated subset to check (default: all 8)
--n_envs        number of parallel environments per task (default: 1)
--no_color      disable ANSI colour output
"""

import sys
import os
import argparse
import traceback

# ── Add experiments/ to path so env imports work ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALL_TASKS = [
    'coiling', 'gathering', 'lifting', 'separation',
    'slingshot', 'unknotting', 'wiring_post', 'wrapping',
]

ENV_MAP = {
    'coiling':    ('envs.env_coiling',    'Train_Env_Coiling'),
    'gathering':  ('envs.env_gathering',  'Train_Env_Gathering'),
    'lifting':    ('envs.env_lifting',    'Train_Env_Lifting'),
    'separation': ('envs.env_separation', 'Train_Env_Separation'),
    'slingshot':  ('envs.env_slingshot',  'Train_Env_Slingshot'),
    'unknotting': ('envs.env_unknotting', 'Train_Env_Unknotting'),
    'wiring_post':('envs.env_wiring_post','Train_Env_Wiring_post'),
    'wrapping':   ('envs.env_wrapping',   'Train_Env_Wrapping'),
}


def _color(text, code):
    return f'\033[{code}m{text}\033[0m'


def verify_task(task, n_envs, use_color):
    import importlib
    from omegaconf import OmegaConf

    mod_name, cls_name = ENV_MAP[task]
    module = importlib.import_module(mod_name)
    EnvClass = getattr(module, cls_name)

    cfg = OmegaConf.create({
        'task': task,
        'n_envs': n_envs,
        'GUI': False,
        'camera': False,
        'raytracer': False,
        'requires_grad': False,
        'log_dir': f'/tmp/dlolab_verify/{task}',
        'n_substeps_per_step': 200,
    })

    env = EnvClass(config=cfg)
    env.reset()
    # Step the scene once to confirm the solver runs
    env.scene.step()
    env.scene.reset()
    return True


def main():
    parser = argparse.ArgumentParser(description='DLO-Lab task verification')
    parser.add_argument('--tasks',    nargs='+', default=ALL_TASKS,
                        choices=ALL_TASKS, metavar='TASK',
                        help='Tasks to verify (default: all 8)')
    parser.add_argument('--n_envs',   type=int, default=1,
                        help='Parallel environments per task (default: 1)')
    parser.add_argument('--no_color', action='store_true',
                        help='Disable ANSI colour output')
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    # Import genesis once — must happen before any env is built
    try:
        import genesis as gs
        if not gs._initialized:
            gs.init(seed=0, precision='64', logging_level='error', backend=gs.gpu, performance_mode=True)
    except Exception as e:
        msg = f'Genesis import/init failed: {e}\nCheck INSTALL.md Step 3.'
        print(_color(msg, '31') if use_color else msg)
        sys.exit(1)

    results = {}
    for task in args.tasks:
        print(f'  Verifying {task:<14}...', end=' ', flush=True)
        try:
            verify_task(task, args.n_envs, use_color)
            tag = _color('[OK]', '32') if use_color else '[OK]'
            print(f'{tag}  scene built, reset + 1 step passed')
            results[task] = True
        except Exception:
            tag = _color('[FAIL]', '31') if use_color else '[FAIL]'
            print(f'{tag}')
            traceback.print_exc()
            results[task] = False

    ok  = [t for t, v in results.items() if v]
    fail = [t for t, v in results.items() if not v]

    print()
    if fail:
        msg = f'{len(ok)}/{len(results)} tasks passed. Failed: {", ".join(fail)}'
        print(_color(msg, '31') if use_color else msg)
        print('See INSTALL.md for common error fixes.')
        sys.exit(1)
    else:
        msg = f'All {len(ok)} tasks verified.'
        print(_color(msg, '32') if use_color else msg)


if __name__ == '__main__':
    main()
