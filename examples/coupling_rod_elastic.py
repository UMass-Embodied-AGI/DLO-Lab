import os
import torch
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
    parser.add_argument("--fov", type=float, default=24.0)
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
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=25,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5, -0.5, -0.1),
            upper_bound=(0.5, 0.5, 0.9),
            grid_density=100,
        ),
        rod_options=gs.options.RODOptions(
            damping=20.0,
            angular_damping=10.0,
            adjacent_gap=2
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=False,
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
            res=(1024, 1024), pos=(-1.4, 1.6, 0.8), up=(0, 0, 1),
            lookat=(0., 0., 0), fov=args.fov, GUI = False
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

    K = 8e5
    E = 1e5
    v1 = scene.add_entity(
        material=gs.materials.ROD.Base(
            segment_radius=0.01,
            segment_mass=0.002,
            K=K,
            E=E,
            use_inextensible=False,
        ),
        morph=gs.morphs.ParameterizedRod(
            type="rod",
            n_vertices=31,
            interval=0.02,
            axis="x",
            pos=(-0.3, 0., 0.2),
            euler=(0.0, 0.0, 90.0),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="dlo-lab/textures/rope01.png",
            ),
            vis_mode='recon',
        ),
    )

    b1 = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=False
        ),
        morph=gs.morphs.Cylinder(
            radius=0.012,
            height=0.3,
            pos=(0, -0.3, 0.15),
            euler=(0, 0, 0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4)
        )
    )

    b2 = scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=False
        ),
        morph=gs.morphs.Cylinder(
            radius=0.012,
            height=0.3,
            pos=(0, 0.3, 0.15),
            euler=(0, 0, 0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 0.4)
        )
    )

    obj_elastic = scene.add_entity(
        material=gs.materials.MPM.Elastic(E=1e4, nu=0.3),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.3),
            size=(0.15, 0.1, 0.07),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.5, 0.5),
            vis_mode='recon',
        ),
    )

    ########################## build ##########################
    scene.build(n_envs=args.n_envs, env_spacing=(2, 2))
    v1.set_fixed_states(
        fixed_ids=[0, 1, 29, 30]
    )

    frames = list()
    for i in range(300):
        scene.step()
        if i % 5 == 0:
            if args.output_folder is not None:
                img = camera.render()[0]
                frames.append(img)

    if args.output_folder is not None:
        os.makedirs(args.output_folder, exist_ok=True)
        ray_traced = f"_raytracer" if args.raytracer else ""
        mediapy.write_video(os.path.join(args.output_folder, f"coupling_rod_elastic{ray_traced}.mp4"), np.array(frames), fps=30)
        gs.logger.info(f"Video saved to {os.path.join(args.output_folder, f'coupling_rod_elastic{ray_traced}.mp4')}")


if __name__ == "__main__":
    main()
