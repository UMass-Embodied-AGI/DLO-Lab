"""Per-task and per-robot configuration for dataset generation."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "coiling":     {"n_additional_obj": 1,  "n_controllers": 1},
    "gathering":   {"n_additional_obj": 3,  "n_controllers": 2},
    "lifting":     {"n_additional_obj": 2,  "n_controllers": 2},
    "separation":  {"n_additional_obj": "dynamic", "n_controllers": 2},
    "slingshot":   {"n_additional_obj": 2,  "n_controllers": 1},
    "unknotting":  {"n_additional_obj": 0,  "n_controllers": 2},
    "wiring_post": {"n_additional_obj": 2,  "n_controllers": 1},
    "wrapping":    {"n_additional_obj": 3,  "n_controllers": 2},
}

# Per-arm dimensions
ARM_STATE_DIM = 8    # joints(7) + gripper_width(1)
ARM_ACTION_DIM = 6   # delta xyz(3) + delta rpy(3)
ARM_JOINT_DIM = 9    # motor joints(7) + finger joints(2)

# Unified dimensions — always bimanual size so all tasks share the same tensor shapes
N_ARMS_UNIFIED      = 2
STATE_DIM_UNIFIED   = N_ARMS_UNIFIED * ARM_STATE_DIM   # 16
ACTION_DIM_UNIFIED  = N_ARMS_UNIFIED * ARM_ACTION_DIM  # 12
JOINT_DIM_UNIFIED   = N_ARMS_UNIFIED * ARM_JOINT_DIM   # 18


# ---------------------------------------------------------------------------
# Environment class lookup
# ---------------------------------------------------------------------------

def get_env_class(task: str):
    """Import and return the environment class for the given task."""
    if task == "coiling":
        from envs.env_coiling import Train_Env_Coiling
        return Train_Env_Coiling
    elif task == "gathering":
        from envs.env_gathering import Train_Env_Gathering
        return Train_Env_Gathering
    elif task == "lifting":
        from envs.env_lifting import Train_Env_Lifting
        return Train_Env_Lifting
    elif task == "separation":
        from envs.env_separation import Train_Env_Separation
        return Train_Env_Separation
    elif task == "slingshot":
        from envs.env_slingshot import Train_Env_Slingshot
        return Train_Env_Slingshot
    elif task == "unknotting":
        from envs.env_unknotting import Train_Env_Unknotting
        return Train_Env_Unknotting
    elif task == "wiring_post":
        from envs.env_wiring_post import Train_Env_Wiring_post
        return Train_Env_Wiring_post
    elif task == "wrapping":
        from envs.env_wrapping import Train_Env_Wrapping
        return Train_Env_Wrapping
    else:
        raise ValueError(f"Unknown task: {task}. Valid: {sorted(TASK_REGISTRY.keys())}")


def get_n_additional_obj(task: str, env) -> int:
    """Return n_additional_obj, resolving dynamic cases."""
    cfg = TASK_REGISTRY[task]
    val = cfg["n_additional_obj"]
    if val == "dynamic":
        return env.rope2.n_vertices
    return val


def get_n_controllers(task: str) -> int:
    return TASK_REGISTRY[task]["n_controllers"]


# ---------------------------------------------------------------------------
# Front camera parameters per task (pos, lookat, up extracted from each env)
# ---------------------------------------------------------------------------

FRONT_CAMERA_PARAMS = {
    "coiling":     {"pos": (0., 0.85, 0.3),  "lookat": (0., 0.0, 0.1),   "up": (0, 0, 1)},
    "gathering":   {"pos": (0.4, 2.0, 0.7),  "lookat": (0.25, 0.0, 0.0),  "up": (0, 0, 1)},
    "lifting":     {"pos": (1.0, 1.7, 1.0),  "lookat": (0.5, 0.0, 0.0),   "up": (0, 0, 1)},
    "separation":  {"pos": (0.4, 0.6, 0.8),  "lookat": (0.4, 0.0, 0.0),   "up": (0, 0, 1)},
    "slingshot":   {"pos": (1.0, -1.4, 1.5), "lookat": (0.12, 0.2, 0.18), "up": (0, 0, 1)},
    "unknotting":  {"pos": (-0.05, -0.2, 0.3),"lookat": (0.7, 0.0, 0.0),  "up": (0, 0, 1)},
    "wiring_post": {"pos": (0.1, 0.6, 0.6),  "lookat": (0.2, 0.2, 0.0),   "up": (0, 0, 1)},
    "wrapping":    {"pos": (1.0, -0.8, 0.8), "lookat": (0.4, 0.0, 0.0),   "up": (0, 0, 1)},
}


# ---------------------------------------------------------------------------
# Camera names
# ---------------------------------------------------------------------------

def get_camera_names(n_controllers: int) -> list[str]:
    """Return unified camera names — always three cameras regardless of arm count.

    Single-arm tasks use wrist_right for the real wrist camera and wrist_left
    is filled with a black frame, keeping the schema consistent across all tasks.
    """
    return ["front", "wrist_right", "wrist_left"]


# ---------------------------------------------------------------------------
# State / action naming
# ---------------------------------------------------------------------------

def _arm_state_names(prefix: str) -> list[str]:
    names = [f"{prefix}_joint_{i+1}" for i in range(7)]
    names += [f"{prefix}_gripper_width"]
    return names


def _arm_action_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_dx", f"{prefix}_dy", f"{prefix}_dz",
        f"{prefix}_droll", f"{prefix}_dpitch", f"{prefix}_dyaw",
    ]


def _arm_joint_action_names(prefix: str) -> list[str]:
    names = [f"{prefix}_joint_{i+1}" for i in range(7)]
    names += [f"{prefix}_finger_1", f"{prefix}_finger_2"]
    return names


def build_state_names(n_controllers: int) -> list[str]:
    """Build named dimensions for observation.state — always unified 16D.

    Single-arm tasks use "right" for the real arm; "left" slots are zero-padded.
    """
    return _arm_state_names("right") + _arm_state_names("left")


def build_action_names(n_controllers: int) -> list[str]:
    """Build named dimensions for action — always unified 12D step_all format.

    Layout: [right_xyz, left_xyz, right_rot, left_rot]
    Single-arm tasks use "right" for the real arm; "left" slots are zero-padded.
    """
    xyz = ["right_dx", "right_dy", "right_dz", "left_dx", "left_dy", "left_dz"]
    rot = ["right_droll", "right_dpitch", "right_dyaw", "left_droll", "left_dpitch", "left_dyaw"]
    return xyz + rot


def build_joint_action_names(n_controllers: int) -> list[str]:
    """Build named dimensions for action.joint — always unified 18D."""
    return _arm_joint_action_names("right") + _arm_joint_action_names("left")
