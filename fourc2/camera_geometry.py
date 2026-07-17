"""Pinhole geometry for MuJoCo cameras and standard optical frames.

MuJoCo camera coordinates use +X right, +Y up and look along -Z.  Standard
optical coordinates use +X right, +Y down and +Z forward.  Keep the only axis
conversion here so callers never scatter sign flips through projection code.

`mujoco.Renderer` converts the OpenGL depth buffer by undoing the perspective
projection.  Its returned metric depth is therefore positive optical Z
(equivalently -Z in MuJoCo camera coordinates), not Euclidean ray length.
"""

from dataclasses import dataclass

import numpy as np


# p_mujoco_camera = MUJOCO_FROM_OPTICAL @ p_optical
MUJOCO_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0])
OPTICAL_FROM_MUJOCO = MUJOCO_FROM_OPTICAL.copy()


@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    matrix: np.ndarray

    @property
    def fx(self):
        return float(self.matrix[0, 0])

    @property
    def fy(self):
        return float(self.matrix[1, 1])

    @property
    def cx(self):
        return float(self.matrix[0, 2])

    @property
    def cy(self):
        return float(self.matrix[1, 2])


def intrinsics_from_fovy(width, height, fovy_degrees):
    """Build square-pixel intrinsics matching MuJoCo's vertical field of view."""
    width, height = int(width), int(height)
    fy = 0.5 * height / np.tan(0.5 * np.deg2rad(float(fovy_degrees)))
    fx = fy
    # OpenGL samples at pixel centres.  For indices 0..W-1 the principal point
    # is therefore (W-1)/2, and likewise for height.
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return PinholeIntrinsics(width, height, matrix)


def camera_world_pose_optical(data, camera_id):
    """Return (position, rotation) with rotation mapping optical to world."""
    position = data.cam_xpos[camera_id].copy()
    world_from_mujoco = data.cam_xmat[camera_id].reshape(3, 3).copy()
    world_from_optical = world_from_mujoco @ MUJOCO_FROM_OPTICAL
    return position, world_from_optical


def homogeneous_transform(rotation, translation):
    """Create T_A_B mapping points from frame B into frame A."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform):
    """Invert a rigid homogeneous transform without a generic matrix inverse."""
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_point(transform, point):
    """Apply T_A_B to one 3D point expressed in frame B."""
    transform = np.asarray(transform, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    return transform[:3, :3] @ point + transform[:3, 3]


def world_from_color_optical_transform(data, camera_id):
    """Read T_world_color_optical directly from the current MuJoCo state."""
    position, world_from_optical = camera_world_pose_optical(data, camera_id)
    return homogeneous_transform(world_from_optical, position)


def world_from_body_transform(data, body_id):
    """Read T_world_body directly from mjData.xpos/xmat."""
    return homogeneous_transform(
        data.xmat[body_id].reshape(3, 3), data.xpos[body_id]
    )


def base_world_transform(data, base_body_id):
    """Return T_base_world using the actual current UR5e base body pose."""
    return invert_transform(world_from_body_transform(data, base_body_id))


def base_from_color_optical_transform(data, camera_id, base_body_id):
    """Return T_base_color_optical = T_base_world @ T_world_color_optical."""
    t_world_color_optical = world_from_color_optical_transform(data, camera_id)
    t_base_world = base_world_transform(data, base_body_id)
    return t_base_world @ t_world_color_optical


def validate_rigid_transform(transform, atol=1e-9):
    """Raise ValueError unless transform is a proper rigid homogeneous matrix."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("rotation determinant is not +1")


def relative_optical_transform(target_pose, source_pose):
    """Return T_target_source such that p_target = T_target_source @ p_source."""
    target_position, world_from_target = target_pose
    source_position, world_from_source = source_pose
    target_from_source = np.eye(4, dtype=np.float64)
    target_from_source[:3, :3] = world_from_target.T @ world_from_source
    target_from_source[:3, 3] = world_from_target.T @ (
        source_position - target_position
    )
    return target_from_source


def unproject_depth(depth, intrinsics):
    """Unproject metric optical-Z depth into an HxWx3 optical point image."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(f"depth shape {depth.shape} does not match intrinsics")
    rows, cols = np.indices(depth.shape, dtype=np.float64)
    points = np.empty(depth.shape + (3,), dtype=np.float64)
    points[..., 0] = (cols - intrinsics.cx) * depth / intrinsics.fx
    points[..., 1] = (rows - intrinsics.cy) * depth / intrinsics.fy
    points[..., 2] = depth
    return points


def transform_points(points, target_from_source):
    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(target_from_source[:3, :3], dtype=np.float64)
    translation = np.asarray(target_from_source[:3, 3], dtype=np.float64)
    return points @ rotation.T + translation


def project_points(points, intrinsics):
    """Project optical-frame points to continuous (u, v) image coordinates."""
    points = np.asarray(points, dtype=np.float64)
    z = points[..., 2]
    u = intrinsics.fx * points[..., 0] / z + intrinsics.cx
    v = intrinsics.fy * points[..., 1] / z + intrinsics.cy
    return np.stack([u, v], axis=-1), z


def align_depth_to_color(depth, depth_intrinsics, color_intrinsics,
                         color_from_depth):
    """Geometrically align depth to color with nearest-pixel z-buffering."""
    point_map, _ = project_depth_points_to_color(
        depth, depth_intrinsics, color_intrinsics, color_from_depth
    )
    aligned = point_map[..., 2].copy()
    aligned[~np.isfinite(aligned)] = 0.0
    return aligned.astype(np.float32)


def project_depth_points_to_color(depth, depth_intrinsics, color_intrinsics,
                                  color_from_depth):
    """Return z-buffered Color-optical XYZ and the number of projected points.

    Each finite positive Depth pixel is unprojected in the Depth optical frame,
    transformed by Color_T_Depth, projected to the Color raster, and resolved
    with a nearest-Z buffer when multiple samples hit the same Color pixel.
    Missing Color pixels contain NaN XYZ, never a fabricated zero point.
    """
    depth = np.asarray(depth, dtype=np.float64)
    depth_points = unproject_depth(depth, depth_intrinsics)
    color_points = transform_points(depth_points, color_from_depth)
    pixels, color_z = project_points(color_points, color_intrinsics)

    u = np.rint(pixels[..., 0]).astype(np.int64)
    v = np.rint(pixels[..., 1]).astype(np.int64)
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(color_z)
        & (color_z > 0.0)
        & (u >= 0)
        & (u < color_intrinsics.width)
        & (v >= 0)
        & (v < color_intrinsics.height)
    )

    flat_indices = (v[valid] * color_intrinsics.width + u[valid]).ravel()
    valid_points = color_points[valid].reshape(-1, 3)
    point_map_flat = np.full(
        (color_intrinsics.width * color_intrinsics.height, 3), np.nan,
        dtype=np.float64,
    )
    if flat_indices.size:
        # Stable depth ordering means the first occurrence of each pixel is the
        # nearest point. This is the XYZ equivalent of a standard z-buffer.
        order = np.argsort(valid_points[:, 2], kind="stable")
        sorted_indices = flat_indices[order]
        _, first = np.unique(sorted_indices, return_index=True)
        winners = order[first]
        point_map_flat[flat_indices[winners]] = valid_points[winners]
    point_map = point_map_flat.reshape(
        color_intrinsics.height, color_intrinsics.width, 3
    )
    return point_map, int(flat_indices.size)
