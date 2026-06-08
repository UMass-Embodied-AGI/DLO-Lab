import os
import argparse
import warnings
import mediapy
import numpy as np
import genesis as gs

warnings.filterwarnings(
    action='ignore',
    message='.*Template mapper caching disabled.*',
    category=UserWarning
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-o", "--output_folder", type=str, default=None)
    parser.add_argument("--fov", type=float, default=30.0)
    parser.add_argument("-n", "--n_envs", type=int, default=1)
    parser.add_argument("-r", "--raytracer", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="64", logging_level="info", backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
        ),
        rod_options=gs.options.RODOptions(
            damping=10.0,
            angular_damping=5.0,
        ),
        vis_options=gs.options.VisOptions(
            plane_reflection=True,
        ),
        show_viewer=args.vis,
        renderer=gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path='dlo-lab/exrs/brown_photostudio_02_4k.exr',
                    image_color=(0.6, 0.6, 0.6),
                    encoding='linear',
                ),
            ),
            env_radius=15.0,
            env_euler=(0, 0, 180),
            lights=[],
        ) if args.raytracer else gs.renderers.Rasterizer(),
    )

    if args.output_folder is not None:
        camera = scene.add_camera(
            res=(1024, 1024), pos=(2, -2, 1.2), up=(0, 0, 1),
            lookat=(0.4, 0., 0), fov=args.fov, GUI = False
        )

    ########################## entities ##########################
    plane = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.1
        ),
        morph=gs.morphs.Plane(
            fixed=True,
            visualization=not args.raytracer,
        ),
    )

    if args.raytracer:
        table = scene.add_entity(
            morph=gs.morphs.Mesh(
                file="dlo-lab/meshes/wooden_table.glb",
                pos=(-0., 0., -0.799418 * 2),
                euler=(0, 0, 0),
                scale=2,
                collision=False,
                fixed=True,
            ),
            surface=gs.surfaces.Default()
        )

    c1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.005,
            segment_mass=0.001,
            K=1e5,
            E=1e5,
            G=1e4,
            plastic_yield=np.inf,
            use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="circle",
            n_vertices=80,
            radius=0.16,
            axis="x",
            pos=(0.65, -0.16, 0.02),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="dlo-lab/textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    franka = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True, coup_friction=0.9
        ),
        morph=gs.morphs.URDF(
            file='urdf/panda_bullet/panda.urdf',
            fixed=True,
            collision=True,
            links_to_keep=['panda_grasptarget'],
        ),
        surface=gs.surfaces.Smooth(),
    )

    gripper_geom_indices = list()
    lf = franka.get_link("panda_leftfinger")
    for gi in lf._geoms:
        gripper_geom_indices.append(gi.idx)
    rf = franka.get_link("panda_rightfinger")
    for gi in rf._geoms:
        gripper_geom_indices.append(gi.idx)
    scene.rod_solver.register_gripper_geom_indices(gripper_geom_indices)

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(2, 2))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)

    # Optional: set control gains
    if args.n_envs == 0:
        franka.set_qpos(np.array([1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]))
    else:
        franka.set_qpos(np.array([[1.56, -0.72, -0.02, -2.09, 0.04, 1.33, 2.4, 0.01, 0.01]] * args.n_envs))
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 80, 80]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 20, 20]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -30, -30]),
        np.array([87, 87, 87, 87, 12, 12, 12, 30, 30]),
    )

    end_effector = franka.get_link("panda_grasptarget")

    # Stage 1: move to pre-grasp pose
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.012]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.012]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    qpos[..., -2:] = 0.02

    franka.set_dofs_position(qpos)

    frames = list()

    # Stage 2: grasp
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_force(
        np.array([-1, -1]) if args.n_envs == 0 else np.array([[-1, -1]] * args.n_envs), fingers_dof
    )  # can also use force control
    for i in range(500):
        scene.step()
        if i % 10 == 0:
            if args.output_folder is not None:
                img = camera.render()[0]
                frames.append(img)

    gs.logger.info("grasped")

    # Stage 3: lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.22]) if args.n_envs == 0 else np.array([[0.65, 0.0, 0.22]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_force(
        np.array([-2, -2]) if args.n_envs == 0 else np.array([[-2, -2]] * args.n_envs), fingers_dof
    )  # can also use force control
    for i in range(400):
        scene.step()
        if i % 10 == 0:
            if args.output_folder is not None:
                img = camera.render()[0]
                frames.append(img)

    # Stage 4: move
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.75, 0.0, 0.22]) if args.n_envs == 0 else np.array([[0.75, 0.0, 0.22]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_force(
        np.array([-2, -2]) if args.n_envs == 0 else np.array([[-2, -2]] * args.n_envs), fingers_dof
    )  # can also use force control
    for i in range(400):
        scene.step()
        if i % 10 == 0:
            if args.output_folder is not None:
                img = camera.render()[0]
                frames.append(img)

    # Stage 5: move
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.75, -0.15, 0.22]) if args.n_envs == 0 else np.array([[0.75, -0.15, 0.22]] * args.n_envs),
        quat=np.array([0, 1, 0, 0]) if args.n_envs == 0 else np.array([[0, 1, 0, 0]] * args.n_envs),
    )
    franka.control_dofs_position(qpos[..., :-2], motors_dof)
    franka.control_dofs_force(
        np.array([-2, -2]) if args.n_envs == 0 else np.array([[-2, -2]] * args.n_envs), fingers_dof
    )  # can also use force control
    for i in range(400):
        scene.step()
        if i % 10 == 0:
            if args.output_folder is not None:
                img = camera.render()[0]
                frames.append(img)

    if args.output_folder is not None:
        os.makedirs(args.output_folder, exist_ok=True)
        ray_traced = f"_raytracer" if args.raytracer else ""
        mediapy.write_video(os.path.join(args.output_folder, f"grasp_rod{ray_traced}.mp4"), np.array(frames), fps=30)
        gs.logger.info(f"Video saved to {os.path.join(args.output_folder, f'grasp_rod{ray_traced}.mp4')}")


if __name__ == "__main__":
    main()
