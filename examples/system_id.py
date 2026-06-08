"""System identification of rod stretching (K) and bending (E) stiffness from masks.

Modes:
  --gen_gt        : run the manipulation script at the *target* (K, E) and save segmentation
                    masks at 6 keyframes.
  --debug_render  : at the target (K, E), overlay the differentiable soft mask against the
                    rasterized GT rope mask to check projection/thickness alignment.
  (default)       : optimization. Starting from an initial (K, E), run the same manipulation,
                    render a soft mask of the rod at each keyframe (Gaussian splat of the
                    rod's tube surface, projected with the camera intrinsics/extrinsics),
                    compare to the GT rope mask via soft-IoU, and descend (K, E) in log space.

Loss & gradient. The segmentation renderer is non-differentiable, so the loss uses a
soft-rasterization of the rod surface instead. Both modes feed the same vector-normalized
momentum optimizer with cosine LR decay.

Usage:
  # 1. Generate the ground-truth masks at the target (K, E).
  python examples/system_id.py --gen_gt -o examples/system_id_gt

  # 2. System identification with reverse-mode autodiff (truncated BPTT window = 20).
  python examples/system_id.py --grad_mode autodiff --bptt_window 20 --n_iters 50 --render_per_iters 10 -o examples/system_id_opt
"""
import os
import math
import argparse
import torch
import warnings
import numpy as np
import genesis as gs
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "experiments"))
from utils.controller import (
    rod_vertex_attached_to_gripper,
    rod_vertex_detached_from_gripper,
    RobotControllerPink,
)
from utils.logging import color_print

warnings.filterwarnings(
    action='ignore',
    message='.*Template mapper caching disabled.*',
    category=UserWarning
)

# ----------------------------------------------------------------------------------------
# Manipulation script (shared by GT generation and optimization so the rope undergoes the
# exact same motion in both).
# ----------------------------------------------------------------------------------------
GT_DIR = os.path.join(os.path.dirname(__file__), "system_id_gt")
N_KEYFRAMES = 6
N_STEPS = 150
OPEN_GAP = 0.0
CLOSE_GAP = 0.048
INIT_QPOS = [-0.679298, 0.418198, 0.037299, 1.016187, -0.012074, 0.597995, -2.198853]
GRASP_VERTS = [15, 16]
DELTA = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.03],
    [0.0, 0.0, 0.03],
    [0.0, 0.0, 0.03],
    [0.0, 0.0, 0.03],
]
GAP = [OPEN_GAP, CLOSE_GAP, CLOSE_GAP, CLOSE_GAP, CLOSE_GAP, CLOSE_GAP, CLOSE_GAP]

# stiffness presets
K_TARGET, E_TARGET = 2e5, 3e6
K_INIT, E_INIT = 1e5, 1e4

PALETTE = np.array([
    [10, 50, 100], [100, 170, 210], [240, 160, 120], [100, 10, 30]
], dtype=np.uint8)


# ----------------------------------------------------------------------------------------
# Differentiable soft-rasterization of the rod tube surface
# ----------------------------------------------------------------------------------------
def camera_matrices_torch(camera, device):
    """Return (K_int (3,3), E_ext (4,4) world->cam) as torch tensors. Camera is static."""
    K_int = torch.as_tensor(np.asarray(camera.intrinsics), dtype=gs.tc_float, device=device)
    E_ext = torch.as_tensor(np.asarray(camera.extrinsics), dtype=gs.tc_float, device=device)
    return K_int, E_ext


def rod_surface_points(state, radius, n_along=4, n_radial=8, env=0):
    """Sample points on the rod's tube surface (differentiable w.r.t. the rod state).

    Mirrors the tube construction in genesis/utils/rod.py:mesh_from_centerline: for each
    edge, place rings ``center + radius*(cos(theta)*d1 + sin(theta)*d2)`` along the edge,
    using the simulator's per-edge material directors d1/d2.

    Returns points of shape (P, 3) in the (offset-free) local sim frame for environment ``env``.
    """
    pos = state.pos[env]          # (V, 3)
    d1 = state.d1[env]            # (E, 3)
    d2 = state.d2[env]            # (E, 3)
    p0 = pos[:-1]                 # (E, 3)
    p1 = pos[1:]                  # (E, 3)

    ts = torch.linspace(0.0, 1.0, n_along, dtype=pos.dtype, device=pos.device)        # (A,)
    centers = p0[:, None, :] * (1.0 - ts)[None, :, None] + p1[:, None, :] * ts[None, :, None]  # (E,A,3)

    thetas = torch.linspace(0.0, 2.0 * math.pi, n_radial + 1, dtype=pos.dtype, device=pos.device)[:-1]  # (R,)
    ring = (torch.cos(thetas)[None, :, None] * d1[:, None, :]
            + torch.sin(thetas)[None, :, None] * d2[:, None, :])  # (E,R,3)

    pts = centers[:, :, None, :] + radius * ring[:, None, :, :]   # (E,A,R,3)
    return pts.reshape(-1, 3)


def project_points(points, K_int, E_ext):
    """Project world points (P,3) to pixel coords. Returns (u, v, valid)."""
    P = points.shape[0]
    ones = torch.ones((P, 1), dtype=points.dtype, device=points.device)
    ph = torch.cat([points, ones], dim=1)          # (P,4)
    Xc = ph @ E_ext.T                              # (P,4) camera frame (OpenCV)
    Xc = Xc[:, :3]
    z = Xc[:, 2]
    uvw = Xc @ K_int.T                             # (P,3)
    zc = z.clamp_min(1e-6)
    u = uvw[:, 0] / zc
    v = uvw[:, 1] / zc
    valid = (z > 1e-4).to(points.dtype)
    return u, v, valid


def splat_soft_mask(u, v, valid, H, W, sigma, win=None):
    """Additive-density soft mask via *windowed* splatting: soft = 1 - exp(-sum_p Gaussian_p).

    Each point only writes a (2*win+1)^2 patch around its projected pixel (scatter via
    index_add), so memory is O(P * win^2) instead of O(P * H * W) -- essential to keep the
    autograd graph small enough to fit in GPU memory. Returns (H,W) in [0,1).
    """
    device = u.device
    if win is None:
        win = int(max(3, math.ceil(3.0 * sigma)))
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    ui = u.detach().round().long()                       # window center (col), detached
    vi = v.detach().round().long()                       # window center (row), detached
    off = torch.arange(-win, win + 1, device=device)
    oy, ox = torch.meshgrid(off, off, indexing="ij")     # (w,w)

    px = ui[:, None, None] + ox[None]                    # (P,w,w) integer cols (u-axis)
    py = vi[:, None, None] + oy[None]                    # (P,w,w) integer rows (v-axis)
    d2 = (px.to(u.dtype) - u[:, None, None]) ** 2 + (py.to(u.dtype) - v[:, None, None]) ** 2
    g = torch.exp(-d2 * inv2s2) * valid[:, None, None]

    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    g = g * inb.to(g.dtype)
    flat_idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
    acc = torch.zeros(H * W, dtype=u.dtype, device=device)
    acc = acc.index_add(0, flat_idx, g.reshape(-1)).reshape(H, W)
    return 1.0 - torch.exp(-acc)


def soft_iou_loss(soft, gt, eps=1e-6):
    inter = (soft * gt).sum()
    union = (soft + gt - soft * gt).sum()
    return 1.0 - inter / (union + eps)


def render_rod_soft_mask(state, radius, K_int, E_ext, H, W, sigma, n_along=4, n_radial=8,
                         env=0, offset=None):
    pts = rod_surface_points(state, radius, n_along=n_along, n_radial=n_radial, env=env)
    if offset is not None:
        # get_state returns offset-free local positions; add the env's world offset so the
        # world-frame camera projection lands correctly (matters for n_envs > 1).
        pts = pts + offset
    u, v, valid = project_points(pts, K_int, E_ext)
    return splat_soft_mask(u, v, valid, H, W, sigma)


# ----------------------------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------------------------
def build_scene(args, requires_grad, K, E, grad_clip=0.0, bptt_window=0):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            segmentation_level='entity'
        ),
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=25,
            requires_grad=requires_grad,
            bptt_window=bptt_window,
        ),
        rod_options=gs.options.RODOptions(
            damping=40.0,
            angular_damping=10.0,
            n_pbd_iters=10,
            grad_clip=grad_clip,
            requires_grad_K=requires_grad,
            requires_grad_E=requires_grad,
        ),
        rigid_options=gs.options.RigidOptions(
            skip_backward=True
        ),
        show_viewer=args.vis,
        renderer=gs.renderers.Rasterizer(),
    )

    camera = scene.add_camera(
        res=(640, 480),
        pos=(0.9147, -0.2669, 0.3031),
        up=(-0.4587, -0.0045, 0.8886),
        lookat=(0.0261, -0.2553, -0.1555),
        fov=64.81,
        GUI=False,
    )

    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.0
        ),
        morph=gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
            visualization=True
        ),
    )

    segment_radius = 0.005
    rope = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=segment_radius,
            segment_mass=0.001,
            K=K,
            E=E,
            G=0,
            use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=32,
            interval=0.02,
            axis="y",
            pos=(0.4, -0.61, 0.005),
            euler=(0, 0, 0),
        ),
        surface=gs.surfaces.Rough(
            color=(0.8, 0.8, 0.8),
            vis_mode='recon'
        ),
    )

    robot = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=1.0
        ),
        morph=gs.morphs.URDF(
            file='urdf/xarm/xarm7_with_gripper_reduced_dof_stl.urdf',
            pos=(0.0, 0.0, 0.0),
            euler=(0., 0., 0.),
            fixed=True,
            collision=True,
            recompute_inertia=True,
            links_to_keep=['link_tcp'],
        ),
        surface=gs.surfaces.Smooth(),
    )

    gripper_geom_indices = []
    for link_name in ("left_finger", "right_finger"):
        for gi in robot.get_link(link_name)._geoms:
            gripper_geom_indices.append(gi.idx)
    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    return scene, camera, plane, rope, robot, segment_radius


def setup_robot(robot, n_envs):
    qpos = np.array([[0, -0.44, 0, 0.96, 0, 1.4, 0, 0, 0]] * n_envs)
    robot.set_qpos(qpos)
    robot.set_dofs_kp(np.array([2500, 2500, 2000, 2000, 1000, 1000, 1000, 150, 150]))
    robot.set_dofs_kv(np.array([800, 800, 600, 600, 400, 400, 400, 60, 60]))
    robot.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -300, -300]),
        np.array([87, 87, 87, 87, 12, 12, 12, 300, 300]),
    )


def run_rollout(scene, rope, robot, ef1, controller, n_envs, on_keyframe):
    """Run the fixed manipulation script. Calls on_keyframe(step_idx) after each keyframe's
    settling steps (just before the next control command)."""
    robot.set_qpos(np.array([INIT_QPOS + [OPEN_GAP, OPEN_GAP]] * n_envs))
    controller.set_initial_dofs_position(torch.tensor(INIT_QPOS + [OPEN_GAP, OPEN_GAP]), False)

    for step in range(len(DELTA)):
        controller.control_robot(
            GAP[step], GAP[step],
            dx=DELTA[step][0], dy=DELTA[step][1], dz=DELTA[step][2],
        )
        for j in range(N_STEPS):
            scene.step()
            if j == N_STEPS - 1:
                if step == 1:
                    for e in range(n_envs):
                        for vid in GRASP_VERTS:
                            rod_vertex_attached_to_gripper(rope, vid, ef1, envs_idx=e)
                on_keyframe(step)


# ----------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------
def gen_gt(args):
    n_envs = 1
    scene, camera, plane, rope, robot, _ = build_scene(args, requires_grad=False, K=K_TARGET, E=E_TARGET)
    scene.build(n_envs=n_envs, env_spacing=(1, 1))
    setup_robot(robot, n_envs)
    ef1 = robot.get_link("link_tcp")
    init_pos = ef1.get_pos()[0]; init_pos[2] = 0.005
    controller = RobotControllerPink(scene, robot, ef1, args, init_pos.tolist(),
                                     initial_gripper_gap=OPEN_GAP, debug=True)

    os.makedirs(args.output_folder, exist_ok=True)
    rope_seg_id = resolve_rope_seg_id(scene, rope)

    def on_keyframe(step):
        mask = camera.render(segmentation=True)[2]
        np.save(os.path.join(args.output_folder, f"system_id_s{step}_c0.npy"), mask)
        Image.fromarray(PALETTE[mask]).save(
            os.path.join(args.output_folder, f"system_id_s{step}_c0.png"))

    run_rollout(scene, rope, robot, ef1, controller, n_envs, on_keyframe)
    color_print(f"GT masks saved to {args.output_folder}. rope_seg_id={rope_seg_id}", "green")


def resolve_rope_seg_id(scene, rope):
    """Return the entity-level segmentation index of the rope (robust, no hard-coding)."""
    seg_map = scene.visualizer.segmentation_idx_dict  # {seg_idxc: seg_key(=entity.idx for 'entity')}
    matches = [idxc for idxc, key in seg_map.items() if key == rope.idx]
    assert len(matches) == 1, f"Could not resolve rope seg id from {seg_map} (rope.idx={rope.idx})"
    return matches[0]


def optimize(args):
    """System identification of (K, E) in log space.

    Two gradient sources (``--grad-mode``):
      fd        : central differences of the loss, evaluated in parallel across 5 overlapping
                  environments in one batched rollout (env0=base, env1/2=K+/-, env3/4=E+/-).
      autodiff  : true reverse-mode gradient w.r.t. K/E (n_envs=1).
    Both feed the same vector-normalized momentum optimizer with cosine LR decay.
    """
    if args.render_per_iters is not None:
        assert args.output_folder is not None, "--render_per_iters requires -o/--output_folder"
        os.makedirs(args.output_folder, exist_ok=True)

    autodiff = args.grad_mode == "autodiff"
    n_envs = 1 if autodiff else 5
    grad_clip = args.grad_clip if autodiff else 0.0
    scene, camera, plane, rope, robot, segment_radius = build_scene(
        args, requires_grad=autodiff, K=K_INIT, E=E_INIT, grad_clip=grad_clip,
        bptt_window=args.bptt_window if autodiff else 0)

    scene.build(n_envs=n_envs, env_spacing=(1, 1) if autodiff else (0, 0))
    setup_robot(robot, n_envs)
    ef1 = robot.get_link("link_tcp")
    init_pos = ef1.get_pos()[0]; init_pos[2] = 0.005
    controller = RobotControllerPink(scene, robot, ef1, args, init_pos.tolist(),
                                     initial_gripper_gap=OPEN_GAP, debug=False)

    device = rope.get_state().pos.device
    K_int, E_ext = camera_matrices_torch(camera, device)
    H, W = 480, 640
    env_offsets = torch.as_tensor(np.asarray(scene.envs_offset), dtype=gs.tc_float, device=device)

    rope_seg_id = resolve_rope_seg_id(scene, rope)
    gt_masks = []
    for k in range(N_KEYFRAMES):
        m = np.load(os.path.join(GT_DIR, f"system_id_s{k}_c0.npy"))
        gt_masks.append(torch.as_tensor((m == rope_seg_id).astype(np.float32),
                                        dtype=gs.tc_float, device=device))
    if args.grad_mode == 'autodiff':
        color_print(f'mode=autodiff grad_clip={grad_clip} bptt_window={args.bptt_window};', "blue")
    elif args.grad_mode == 'fd':
        color_print(f'mode=fd fd_eps={args.fd_eps};', "blue")
    color_print(f"loaded {len(gt_masks)} GT masks; rope_seg_id={rope_seg_id}", "blue", flush=True)

    eps = args.fd_eps  # finite-difference step in log10 space
    ln10 = math.log(10.0)

    def _rollout_keyframes(Ks, Es):
        """Reset, set per-env (K,E), run the rollout, return {keyframe: RODEntityState}."""
        scene.reset()
        for e in range(n_envs):
            for vid in GRASP_VERTS:
                rod_vertex_detached_from_gripper(rope, vid, envs_idx=e)
        rope.set_stretching_stiffness(Ks)
        rope.set_bending_stiffness(Es)
        ks = {}

        def on_keyframe(step):
            ks[step] = rope.get_state()

        run_rollout(scene, rope, robot, ef1, controller, n_envs, on_keyframe)
        return ks

    @torch.no_grad()
    def save_overlay(tag, logK_val, logE_val):
        """Render the rope at the current (K, E) and overlay its soft mask (red) on the GT rope
        mask (green; overlap shows yellow) at every keyframe, saved as one montage PNG."""
        K_val, E_val = 10.0 ** logK_val, 10.0 ** logE_val
        ks = _rollout_keyframes(torch.full((n_envs,), K_val, dtype=gs.tc_float),
                                torch.full((n_envs,), E_val, dtype=gs.tc_float))
        panels = []
        for k in range(N_KEYFRAMES):
            soft = render_rod_soft_mask(ks[k], segment_radius, K_int, E_ext, H, W, args.sigma,
                                        n_along=args.n_along, n_radial=args.n_radial,
                                        env=0, offset=env_offsets[0])
            pred = soft.detach().cpu().numpy() > 0.5
            gt = gt_masks[k].detach().cpu().numpy() > 0.5
            rgb = np.zeros((H, W, 3), np.uint8)
            rgb[gt] = [0, 200, 0]                              # GT rope -> green
            rgb[pred] += np.array([200, 0, 0], np.uint8)       # current rope -> red (overlap=yellow)
            panels.append(rgb)
        path = os.path.join(args.output_folder, f"opt_overlay_{tag}.png")
        Image.fromarray(np.concatenate(panels, axis=1)).save(path)
        color_print(f"  [render] K={K_val:.4e} E={E_val:.4e} overlay -> {path}", "green", flush=True)

    @torch.no_grad()
    def grad_fd(logK_val, logE_val):
        """Batched central-difference gradient (5 envs). Returns (base_loss, dL/dlogK, dL/dlogE)."""
        logK_envs = np.array([logK_val, logK_val + eps, logK_val - eps, logK_val, logK_val])
        logE_envs = np.array([logE_val, logE_val, logE_val, logE_val + eps, logE_val - eps])
        ks = _rollout_keyframes(torch.as_tensor(10.0 ** logK_envs, dtype=gs.tc_float),
                                torch.as_tensor(10.0 ** logE_envs, dtype=gs.tc_float))
        losses = torch.zeros(n_envs, dtype=gs.tc_float, device=device)
        for k in range(N_KEYFRAMES):
            for e in range(n_envs):
                soft = render_rod_soft_mask(ks[k], segment_radius, K_int, E_ext, H, W, args.sigma,
                                            n_along=args.n_along, n_radial=args.n_radial,
                                            env=e, offset=env_offsets[e])
                losses[e] = losses[e] + soft_iou_loss(soft, gt_masks[k])
        l = losses.cpu().numpy()
        return float(l[0]), (l[1] - l[2]) / (2 * eps), (l[3] - l[4]) / (2 * eps)

    def grad_autodiff(logK_val, logE_val):
        """Reverse-mode gradient (clipped). Returns (base_loss, dL/dlogK, dL/dlogE)."""
        K_val, E_val = 10.0 ** logK_val, 10.0 ** logE_val
        ks = _rollout_keyframes(torch.full((n_envs,), K_val, dtype=gs.tc_float),
                                torch.full((n_envs,), E_val, dtype=gs.tc_float))
        loss = torch.zeros((), dtype=gs.tc_float, device=device)
        for k in range(N_KEYFRAMES):
            soft = render_rod_soft_mask(ks[k], segment_radius, K_int, E_ext, H, W, args.sigma,
                                        n_along=args.n_along, n_radial=args.n_radial, env=0)
            loss = loss + soft_iou_loss(soft, gt_masks[k])
        loss.backward()
        if getattr(loss, "scene", None) is None:
            scene._backward()
        gK = float(rope.get_stretching_stiffness_grad()[0]) * K_val * ln10
        gE = float(rope.get_bending_stiffness_grad()[0]) * E_val * ln10
        return float(loss.item()), gK, gE

    grad_fn = grad_autodiff if autodiff else grad_fd

    logK = math.log10(args.init_K)
    logE = math.log10(args.init_E)
    # Momentum on the *vector-normalized* gradient. Unlike per-coordinate Adam, this preserves
    # the relative magnitude of the two parameters' gradients, so a weakly-observable parameter
    # (here K, barely tensioned by the light rope) takes proportionally tiny steps instead of
    # being amplified to a full-size, noise-driven step.
    mom = np.zeros(2)
    beta = 0.5

    for it in range(args.n_iters):
        base, gK, gE = grad_fn(logK, logE)   # dL/dlogK, dL/dlogE
        if args.render_per_iters is not None and it % args.render_per_iters == 0:
            save_overlay(f"it{it:03d}", logK, logE)  # params evaluated this iter (match `base` loss)
        g = np.array([gK, gE])
        g = g / (np.linalg.norm(g) + 1e-12)   # unit direction (preserves K/E ratio)
        mom = beta * mom + (1 - beta) * g
        # cosine learning-rate decay so the step settles inside the low-loss basin
        lr_t = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * it / max(1, args.n_iters - 1)))
        logK = float(np.clip(logK - lr_t * mom[0], 3.0, 8.0))
        logE = float(np.clip(logE - lr_t * mom[1], 2.0, 8.0))

        print(f"[it {it:03d}] loss={base:.5f}  K={10.0**logK:.4e}  E={10.0**logE:.4e}  "
              f"gK={gK:+.4f} gE={gE:+.4f} lr={lr_t:.3f}  | target K={K_TARGET:.2e} E={E_TARGET:.2e}", flush=True)

    if args.render_per_iters is not None:
        save_overlay("final", logK, logE)
    print(f"Final: K={10.0**logK:.4e}  E={10.0**logE:.4e}  (target K={K_TARGET:.2e} E={E_TARGET:.2e})")


def debug_render(args):
    """Fast (no-grad) diagnostic: run at target params, and at each keyframe compare the
    differentiable soft mask against the rasterized GT rope mask. Saves overlays + stats."""
    n_envs = 1
    scene, camera, plane, rope, robot, segment_radius = build_scene(
        args, requires_grad=False, K=K_TARGET, E=E_TARGET)
    scene.build(n_envs=n_envs, env_spacing=(1, 1))
    setup_robot(robot, n_envs)
    ef1 = robot.get_link("link_tcp")
    init_pos = ef1.get_pos()[0]; init_pos[2] = 0.005
    controller = RobotControllerPink(scene, robot, ef1, args, init_pos.tolist(),
                                     initial_gripper_gap=OPEN_GAP, debug=False)
    device = rope.get_state().pos.device
    K_int, E_ext = camera_matrices_torch(camera, device)
    H, W = 480, 640
    rope_seg_id = resolve_rope_seg_id(scene, rope)
    out = args.output_folder or "/tmp/sysid_debug"
    os.makedirs(out, exist_ok=True)

    sigmas = [0.6, 0.8, 1.0, 1.4, 2.0, 3.0]

    def on_keyframe(step):
        seg = camera.render(segmentation=True)[2]
        gt = (seg == rope_seg_id)
        gt_t = torch.as_tensor(gt.astype(np.float32), dtype=gs.tc_float, device=device)
        state = rope.get_state()
        line = f"s{step}: gt_px={int(gt.sum())} | "
        best = None
        for sg in sigmas:
            soft = render_rod_soft_mask(state, segment_radius, K_int, E_ext, H, W, sg,
                                        n_along=args.n_along, n_radial=args.n_radial)
            siou = soft_iou_loss(soft, gt_t).item()
            s = soft.detach().cpu().numpy()
            pred = s > 0.5
            inter = (pred & gt).sum(); union = (pred | gt).sum()
            iou = inter / max(int(union), 1)
            line += f"[s={sg}: loss={siou:.3f} IoU={iou:.2f} px={int(pred.sum())}] "
            if best is None or siou < best[1]:
                best = (sg, siou, pred)
        print(line)
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[gt] = [0, 200, 0]
        rgb[best[2]] += np.array([200, 0, 0], np.uint8)
        Image.fromarray(rgb).save(os.path.join(out, f"overlay_s{step}.png"))

    run_rollout(scene, rope, robot, ef1, controller, n_envs, on_keyframe)
    color_print(f"Overlays saved to {out}", "green")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-o", "--output_folder", type=str, default=None)
    parser.add_argument("--gen_gt", action="store_true", default=False,
                        help="Regenerate GT masks at the target (K,E) instead of optimizing.")
    parser.add_argument("--debug_render", action="store_true", default=False,
                        help="Diagnose soft-mask vs GT-mask alignment at target params.")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="per-substep adjoint clip threshold for autodiff (0 disables)")
    parser.add_argument("--bptt_window", type=int, default=20,
                        help="truncated-BPTT window in steps for autodiff (0 = full BPTT)")
    parser.add_argument("--grad_mode", type=str, default="autodiff", choices=["fd", "autodiff"],
                        help="gradient source for optimization: batched finite differences or "
                             "clipped reverse-mode autodiff")
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0, help="splat Gaussian std in pixels")
    parser.add_argument("--n_along", type=int, default=4)
    parser.add_argument("--n_radial", type=int, default=8)
    parser.add_argument("--init_K", type=float, default=K_INIT)
    parser.add_argument("--init_E", type=float, default=E_INIT)
    parser.add_argument("--fd_eps", type=float, default=0.1,
                        help="finite-difference step in log10 space for the batched gradient")
    parser.add_argument("--lr_min", type=float, default=0.02,
                        help="final (decayed) learning rate in log10 space")
    parser.add_argument("--render_per_iters", type=int, default=None,
                        help="If set, every N optimization iterations render the current rope from "
                             "the camera view, overlay its soft mask on the GT rope mask, and save a "
                             "montage PNG to --output_folder (also renders the final result).")
    args = parser.parse_args()

    gs.init(seed=0, precision="64", logging_level="error", backend=gs.gpu, performance_mode=True)

    if args.gen_gt:
        assert args.output_folder is not None, "--gen-gt requires -o/--output_folder"
        gen_gt(args)
    elif args.debug_render:
        debug_render(args)
    else:
        optimize(args)


if __name__ == "__main__":
    main()
