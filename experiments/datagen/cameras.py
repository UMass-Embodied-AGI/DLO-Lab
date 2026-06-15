"""Camera setup and rendering helpers for LeRobot dataset generation.

This is the only datagen module directly coupled to the Genesis camera API.
Camera poses (FRONT_CAMERA_PARAMS in config.py) and the wrist offset
transform (WRIST_OFFSET_T) were tuned for the original scene layouts and
may need visual re-tuning if env scenes change.
"""

from __future__ import annotations

import numpy as np
import torch

from datagen.config import FRONT_CAMERA_PARAMS


def make_construct_extra_cameras(
    task: str, n_controllers: int,
    img_resolution: tuple[int, int],
    wrist_resolution: tuple[int, int],
):
    """Return a construct_extra_cameras method that adds front + wrist cameras.

    Creates all datagen cameras via add_camera() before scene.build().
    The wrist cameras are attached to robot links after build via attach_wrist_cameras().
    We pass camera=False to the env so the default high-res cameras are not created.
    """
    front_params = FRONT_CAMERA_PARAMS[task]

    def construct_extra_cameras(self):
        # Front camera with same viewpoint as env but at datagen resolution
        front = self.scene.add_camera(
            res=img_resolution,
            pos=front_params["pos"],
            lookat=front_params["lookat"],
            up=front_params["up"],
            fov=65,
            GUI=False,
        )
        self._datagen_cameras = {"front": front}

        # Wrist cameras — always produce wrist_right + wrist_left for unified schema.
        # For single-arm tasks wrist_left is a black numpy frame (no physical camera).
        cam_r = self.scene.add_camera(res=wrist_resolution, fov=90, GUI=False)
        self._datagen_cameras["wrist_right"] = cam_r
        if n_controllers >= 2:
            cam_l = self.scene.add_camera(res=wrist_resolution, fov=90, GUI=False)
            self._datagen_cameras["wrist_left"] = cam_l
        else:
            # Black frame placeholder stored as a numpy array (not a camera object)
            H, W = wrist_resolution[1], wrist_resolution[0]
            self._datagen_cameras["wrist_left"] = np.zeros((H, W, 3), dtype=np.uint8)

    return construct_extra_cameras


# Wrist camera offset transform (camera_link → camera world pose).
# Base rotation: R0 = Ry(-pi/2) * Rz(90°) = [[0,0,-1],[1,0,0],[0,-1,0]]
# Pitch down 15° in camera space: R = R0 @ Rx(-15°), tilts look direction toward workspace.
# tx=-0.09: lift camera 9cm upward along camera_link X (= panda_hand Z = down axis)
# tz=-0.20: 20cm along camera_link Z (= panda_hand X = horizontal), closer to rope tip
_s15 = float(np.sin(np.radians(15)))   # ≈ 0.259
_c15 = float(np.cos(np.radians(15)))   # ≈ 0.966
WRIST_OFFSET_T = np.array(
    [
        [ 0,  _s15, -_c15, -0.08],
        [ 1,  0,     0,     0   ],
        [ 0, -_c15, -_s15, -0.22],
        [ 0,  0,     0,     1   ],
    ],
    dtype=np.float32,
)


def attach_wrist_cameras(env, n_controllers: int):
    """Attach wrist cameras to robot links after scene.build().

    Must be called after env construction (scene is already built).
    Uses camera.attach(link, WRIST_OFFSET_T) + camera.move_to_attach().
    For single-arm tasks wrist_left is a numpy black frame — no attachment needed.
    """
    cams = getattr(env, "_datagen_cameras", {})
    link1 = env.franka1.get_link("camera_link")
    cams["wrist_right"].attach(link1, WRIST_OFFSET_T)
    cams["wrist_right"].move_to_attach()

    if n_controllers >= 2:
        link2 = env.franka2.get_link("camera_link")
        cams["wrist_left"].attach(link2, WRIST_OFFSET_T)
        cams["wrist_left"].move_to_attach()
    # else: wrist_left is a numpy black frame, no attachment needed


def collect_datagen_cameras(env) -> dict:
    """Collect all datagen cameras created by construct_extra_cameras hook.

    Must be called after env construction.
    """
    cameras = getattr(env, "_datagen_cameras", {})
    if not cameras:
        raise RuntimeError(
            "No datagen cameras found. Ensure construct_extra_cameras was monkey-patched."
        )
    return cameras


def move_camera_to_env(
    cam,
    env_idx: int,
    env_offset: np.ndarray,
    front_params: dict | None,
    offset_T: np.ndarray | None,
):
    """Move camera to the world position for env_idx before rendering.

    For free (front) cameras: shifts base pos/lookat by env_offset.
    For attached (wrist) cameras: computes link world transform for env_idx
    and applies offset_T, since move_to_attach() is hardcoded to env 0.
    """
    if getattr(cam, "_attached_link", None) is not None:
        from genesis.utils.geom import quat_to_R
        link = cam._attached_link
        pos  = link.get_pos()[env_idx].cpu().numpy()   # (3,)
        quat = link.get_quat()[env_idx].cpu().numpy()  # (4,) wxyz
        R = np.array(quat_to_R(quat), dtype=np.float32)
        link_T = np.eye(4, dtype=np.float32)
        link_T[:3, :3] = R
        link_T[:3,  3] = pos
        cam_T = (link_T @ offset_T).astype(np.float32)
        cam.set_pose(transform=cam_T)
    else:
        base_pos    = np.array(front_params["pos"],    dtype=np.float64)
        base_lookat = np.array(front_params["lookat"], dtype=np.float64)
        base_up     = np.array(front_params["up"],     dtype=np.float64)
        cam.set_pose(
            pos    = base_pos    + env_offset,
            lookat = base_lookat + env_offset,
            up     = base_up,
        )


def render_camera(
    cam,
    env_idx: int,
    env_offset: np.ndarray,
    front_params: dict | None,
    offset_T: np.ndarray | None,
) -> np.ndarray:
    """Move camera to env_idx's world position, then render (H, W, 3) uint8."""
    move_camera_to_env(cam, env_idx, env_offset, front_params, offset_T)
    img = cam.render()[0]
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img is not None and img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def capture_images(
    datagen_cameras: dict[str, object],
    camera_names: list[str],
    env_idx: int,
    env_offset: np.ndarray,
    front_params: dict,
) -> dict[str, np.ndarray]:
    """Render all cameras for env_idx and return {name: (H,W,3) uint8 array}.

    Moves each camera to the world position for env_idx before rendering.
    Free (front) cameras are offset by env_offset; attached (wrist) cameras
    use the link's world transform for env_idx composed with WRIST_OFFSET_T.
    Numpy array entries (e.g. wrist_left black frame for single-arm tasks) are
    returned as-is without rendering.
    """
    images = {}
    for cam_name in camera_names:
        cam = datagen_cameras.get(cam_name)
        if cam is None:
            continue
        elif isinstance(cam, np.ndarray):
            images[cam_name] = cam   # pre-generated black frame
        else:
            is_attached = getattr(cam, "_attached_link", None) is not None
            images[cam_name] = render_camera(
                cam, env_idx, env_offset,
                front_params=None if is_attached else front_params,
                offset_T=WRIST_OFFSET_T if is_attached else None,
            )
    return images
