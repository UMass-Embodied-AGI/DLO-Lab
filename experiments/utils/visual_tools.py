import cv2
import torch
import numpy as np


# Default color palette with distinct, visually pleasing colors (RGB format)
DEFAULT_COLOR_PALETTE = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    # (255, 128, 0),    # Orange
    # (128, 0, 255),    # Purple
    # (0, 255, 128),    # Spring Green
    # (255, 0, 128),    # Pink
    # (128, 255, 0),    # Lime
    # (0, 128, 255),    # Sky Blue
    # (255, 128, 128),  # Light Red
    # (128, 255, 128),  # Light Green
    # (128, 128, 255),  # Light Blue
    # (255, 255, 128),  # Light Yellow
]


def draw_points(
    camera, points_3d, rgb_img=None, color=(0, 255, 0), alpha=1.0, radius=5, thickness=-1,
    annotation_args=None, text_offset=(10, -10), text_color=None, text_scale=0.5, text_thickness=1,
    use_color_palette=False, color_palette=None, return_points_2d=False
):
    """
    Project 3D world points to 2D image coordinates and draw them on the rendered image.

    Parameters
    ----------
    camera : genesis.Camera
        The camera object used for rendering and projection.
    points_3d : np.ndarray or list
        3D points in world coordinates. Shape (N, 3) or list of (x, y, z) tuples.
    rgb_img : np.ndarray, optional
        RGB image to draw on. If None, will render from camera. Shape (H, W, 3).
    color : tuple, optional
        RGB color for the points (3 values). Ignored if use_color_palette=True.
        Default is green (0, 255, 0).
    alpha : float, optional
        Transparency level in range [0, 1]. 0 is fully transparent, 1 is fully opaque.
        Default is 1.0 (fully opaque).
    radius : int, optional
        Radius of the circles to draw. Default is 5.
    thickness : int, optional
        Thickness of the circle. -1 means filled. Default is -1.
    annotation_args : list of str, optional
        List of text annotations for each point. Length must match number of points.
        If None, no annotations are drawn.
    text_offset : tuple of int, optional
        2D offset (dx, dy) in pixels for text placement relative to each point.
        Default is (10, -10).
    text_color : tuple, optional
        RGB color for the text (3 values). If None, uses same color as points.
        Ignored if use_color_palette=True. Default is None.
    text_scale : float, optional
        Font scale for the text. Default is 0.5.
    text_thickness : int, optional
        Thickness of the text. Default is 1.
    use_color_palette : bool, optional
        If True, automatically assign colors from a palette to each point.
        This overrides the 'color' and 'text_color' parameters. Default is False.
    color_palette : list of tuples, optional
        Custom color palette to use. Each tuple should be RGB (3 values).
        If None, uses DEFAULT_COLOR_PALETTE. Colors cycle if more points than colors.
        Default is None.
    return_points_2d : bool, optional
        If True, also return the projected 2D points. Default is False.

    Returns
    -------
    np.ndarray
        Image with drawn points, shape (H, W, 3), dtype uint8 in RGB format.
    2D points : list of tuples, optional
        If return_points_2d is True, also returns a list of (u, v) tuples for projected 2D points.
    """
    # Convert points to numpy array if needed
    if isinstance(points_3d, list):
        points_3d = np.array(points_3d, dtype=np.float32)
    else:
        points_3d = np.asarray(points_3d, dtype=np.float32)

    # Ensure shape is (N, 3)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, 3)
    assert points_3d.shape[1] == 3, f"Expected points shape (N, 3), got {points_3d.shape}"

    n_points = len(points_3d)

    # Validate annotation_args if provided
    if annotation_args is not None:
        if not isinstance(annotation_args, (list, tuple)):
            raise ValueError(f"annotation_args must be a list or tuple, got {type(annotation_args)}")
        if len(annotation_args) != n_points:
            raise ValueError(f"annotation_args length ({len(annotation_args)}) must match number of points ({n_points})")

    # Setup color palette if requested
    if use_color_palette:
        if color_palette is None:
            palette = DEFAULT_COLOR_PALETTE
        else:
            # Validate custom palette
            if not isinstance(color_palette, (list, tuple)):
                raise ValueError(f"color_palette must be a list or tuple, got {type(color_palette)}")
            for i, c in enumerate(color_palette):
                if len(c) != 3:
                    raise ValueError(f"color_palette[{i}] must be RGB (3 values), got {len(c)} values")
            palette = color_palette

        # Create color list for each point (cycling through palette)
        point_colors = [palette[i % len(palette)] for i in range(n_points)]
    else:
        # Validate single color
        if len(color) != 3:
            raise ValueError(f"Color must be RGB (3 values), got {len(color)} values")
        # Use the same color for all points
        point_colors = [tuple(color)] * n_points

    alpha = np.clip(alpha, 0.0, 1.0)  # Clamp to valid range
    use_alpha = alpha < 1.0

    # Set text colors
    if use_color_palette:
        # When using palette, text color matches point color for each point
        text_colors = point_colors
    elif text_color is None:
        # Use same color as points for all text
        text_colors = [point_colors[0]] * n_points
    else:
        # Use custom text color for all text
        if len(text_color) != 3:
            raise ValueError(f"text_color must be RGB (3 values), got {len(text_color)} values")
        text_colors = [tuple(text_color)] * n_points

    # Render image if not provided
    if rgb_img is None:
        rgb_img, _, _, _ = camera.render(rgb=True)
        # Convert tensor to numpy if needed
        if torch.is_tensor(rgb_img):
            rgb_img = rgb_img.cpu().numpy()

    # Handle batched images (take first if batched)
    if rgb_img.ndim == 4:
        rgb_img = rgb_img[0]

    # Ensure image is uint8 and copy for drawing
    if rgb_img.dtype != np.uint8:
        # Assume float in [0, 1] range
        img_draw = (rgb_img * 255).astype(np.uint8)
    else:
        img_draw = rgb_img.copy()

    # Convert RGB to BGR for OpenCV
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

    # Convert all point colors and text colors to BGR
    point_bgr_colors = [(c[2], c[1], c[0]) for c in point_colors]
    text_bgr_colors = [(c[2], c[1], c[0]) for c in text_colors]

    # Get camera matrices
    # The camera uses OpenGL convention, need to convert to OpenCV
    T_OPENGL_TO_OPENCV = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)

    # Camera transform in OpenGL convention
    cam_transform = camera.transform  # (4, 4)

    # Convert to OpenCV convention and compute extrinsics
    cam_pose_opencv = cam_transform @ T_OPENGL_TO_OPENCV
    extrinsics = np.linalg.inv(cam_pose_opencv)  # World to camera

    # Get intrinsics
    intrinsics = camera.intrinsics  # (3, 3)

    # Convert 3D points to homogeneous coordinates
    points_3d_hom = np.concatenate([points_3d, np.ones((n_points, 1))], axis=1)  # (N, 4)

    # Transform to camera coordinates
    points_cam = (extrinsics @ points_3d_hom.T).T  # (N, 4)

    # If using alpha blending, create an overlay
    if use_alpha and alpha < 1.0:
        overlay = img_draw.copy()
    else:
        overlay = img_draw

    # Project to image coordinates
    points_2d_list = []
    for i in range(n_points):
        if points_cam[i, 2] > 0:  # Check if point is in front of camera
            # Project using intrinsics
            p_2d_hom = intrinsics @ points_cam[i, :3]
            p_2d = p_2d_hom[:2] / p_2d_hom[2]

            # Convert to integer pixel coordinates
            u, v = int(round(p_2d[0])), int(round(p_2d[1]))

            # Check if within image bounds
            h, w = img_draw.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                points_2d_list.append((u, v))

                # Get color for this specific point
                point_bgr = point_bgr_colors[i]
                text_bgr = text_bgr_colors[i]

                # Draw circle at the projected point
                cv2.circle(overlay, (u, v), radius, point_bgr, thickness)

                # Draw text annotation if provided
                if annotation_args is not None:
                    text = str(annotation_args[i])
                    text_x = u + text_offset[0]
                    text_y = v + text_offset[1]

                    # Draw text (using the overlay to respect alpha blending)
                    cv2.putText(
                        overlay,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        text_bgr,
                        text_thickness,
                        cv2.LINE_AA
                    )

    # Apply alpha blending if needed
    if use_alpha and alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img_draw, 1 - alpha, 0, img_draw)

    # Convert back to RGB
    cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB, img_draw)

    if return_points_2d:
        return img_draw, points_2d_list

    return img_draw
