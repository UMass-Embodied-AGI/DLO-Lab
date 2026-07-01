"""
Central registry mapping task-name strings to their ``Train_Env`` subclasses.

Every optimizer entry script shares this single mapping instead of keeping its own copy:

    RL      -> experiments/rl/rudinppo.py, experiments/rl/sac.py
    DiffRL  -> experiments/rl/shac.py
    CMA-ES  -> experiments/trajopt/cmaes.py
    GD      -> experiments/trajopt/gd.py

To add a new task, register it here once and every entry script picks it up.
"""

from typing import Dict, Type

from envs.base import Train_Env
from envs.env_coiling import Train_Env_Coiling
from envs.env_gathering import Train_Env_Gathering
from envs.env_lifting import Train_Env_Lifting
from envs.env_separation import Train_Env_Separation
from envs.env_slingshot import Train_Env_Slingshot
from envs.env_unknotting import Train_Env_Unknotting
from envs.env_wiring_post import Train_Env_Wiring_post
from envs.env_wrapping import Train_Env_Wrapping


ENV_REGISTRY: Dict[str, Type[Train_Env]] = {
    "coiling": Train_Env_Coiling,
    "gathering": Train_Env_Gathering,
    "lifting": Train_Env_Lifting,
    "separation": Train_Env_Separation,
    "slingshot": Train_Env_Slingshot,
    "unknotting": Train_Env_Unknotting,
    "wiring_post": Train_Env_Wiring_post,
    "wrapping": Train_Env_Wrapping,
}


def get_env_class(task: str) -> Type[Train_Env]:
    """Return the ``Train_Env`` subclass registered for ``task``.

    Lookup is case-insensitive. Raises ``ValueError`` listing the valid tasks when
    ``task`` is not registered.
    """
    key = task.lower()
    if key not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown task '{task}'. Valid: {sorted(ENV_REGISTRY.keys())}"
        )
    return ENV_REGISTRY[key]
