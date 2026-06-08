coiling_args = {
    'pos_bound': (-0.05, -0.01, 0.05, 0.01),
    # 'mass_list': [0.001, 0.002, 0.005, 0.0075, 0.01],
    # 'radius_list': [0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e2, 5e4),
    # 'twisting_stiffness_bound': (5e2, 5e4),
}

gathering_args = {
    # 'pos_bound': (-0.05, 0., 0.05, 0.05),
    # 'mass_list': [0.01, 0.015, 0.02],
    # 'radius_list': [0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e2, 5e4),
    # 'stretching_stiffness_bound': (5e4, 5e5),
    # 'twisting_stiffness_bound': (5e2, 5e4),
}

lifting_args = {
    'pos_bound': (-0.05, -0.025, 0.05, 0.025),
    # 'mass_list': [0.001, 0.002, 0.005, 0.0075, 0.01],
    # 'radius_list': [0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e2, 5e4),
    # 'stretching_stiffness_bound': (5e4, 5e5),
    # 'twisting_stiffness_bound': (5e2, 5e4),
}

separation_args = {
    'pos_bound': (-0.05, -0.025, 0.05, 0.025),
    # 'mass_list': [0.001, 0.002, 0.005, 0.0075, 0.01],
    # 'radius_list': [0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e2, 5e4),
    # 'twisting_stiffness_bound': (5e2, 5e4),
    # 'friction_list': [[0.3, 0.25], [0.6, 0.5], [0.9, 0.75]],  # (mu_s, mu_k) pairs
}

slingshot_args = {
    # 'mass_list': [0.001, 0.002, 0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e4, 5e5),
    # 'stretching_stiffness_bound': (5e5, 1e6),
}

wrapping_args = {
    'pos_bound': (-0.025, -0.01, 0.025, 0.01),
    # 'mass_list': [0.001, 0.002, 0.005, 0.0075, 0.01],
    # 'radius_list': [0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e3, 5e4),
    # 'stretching_stiffness_bound': (5e4, 5e5),
}

unknotting_args = {
    'pos_bound': (-0.05, -0.025, 0.05, 0.025),
    # 'mass_list': [0.001, 0.002, 0.005, 0.0075, 0.01],
    # 'radius_list': [0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e3, 5e4),
    # 'stretching_stiffness_bound': (5e4, 5e5),
}

wiring_post_args = {
    'pos_bound': (-0.05, -0.025, 0.05, 0.025),
    # 'mass_list': [0.001, 0.002, 0.005, 0.0075, 0.01],
    # 'radius_list': [0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e3, 5e4),
    # 'twisting_stiffness_bound': (5e2, 5e3),
}

wiring_ring_real_args = {
    'pos_bound': (-0.05, -0.025, 0.05, 0.025),
    # 'mass_list': [0.001, 0.002, 0.005, 0.0075, 0.01],
    # 'radius_list': [0.005, 0.0075, 0.01],
    # 'bending_stiffness_bound': (5e3, 5e4),
    # 'twisting_stiffness_bound': (5e2, 5e3),
}

randomized_config = {
    'coiling': coiling_args,
    'gathering': gathering_args,
    'lifting': lifting_args,
    'separation': separation_args,
    'slingshot': slingshot_args,
    'wrapping': wrapping_args,
    'unknotting': unknotting_args,
    'wiring_post': wiring_post_args,
}
