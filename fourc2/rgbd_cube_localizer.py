"""HSV + projected RGB-D localization for the configured red 3 cm cube."""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from fourc2.camera_geometry import project_depth_points_to_color


@dataclass
class CubeLocalizationResult:
    valid: bool
    failure_reason: str | None
    hsv_detected: bool
    contour_area_px: float | None
    mask_pixel_count: int
    eroded_mask_pixel_count: int
    projected_depth_point_count: int
    mask_depth_point_count: int
    filtered_depth_point_count: int
    valid_point_ratio: float
    depth_median_m: float | None
    depth_mad_m: float | None
    bbox_xywh: tuple | None
    pixel_center_uv: tuple | None
    visible_surface_point_color: np.ndarray | None
    estimated_object_center_color: np.ndarray | None
    mask: np.ndarray
    eroded_mask: np.ndarray
    projected_point_map: np.ndarray

    def to_dict(self):
        def array_or_none(value):
            return None if value is None else np.asarray(value).tolist()

        return {
            "valid": self.valid,
            "failure_reason": self.failure_reason,
            "hsv_detected": self.hsv_detected,
            "contour_area_px": self.contour_area_px,
            "mask_pixel_count": self.mask_pixel_count,
            "eroded_mask_pixel_count": self.eroded_mask_pixel_count,
            "projected_depth_point_count": self.projected_depth_point_count,
            "mask_depth_point_count": self.mask_depth_point_count,
            "filtered_depth_point_count": self.filtered_depth_point_count,
            "valid_point_ratio": self.valid_point_ratio,
            "depth_median_m": self.depth_median_m,
            "depth_mad_m": self.depth_mad_m,
            "bbox_xywh": self.bbox_xywh,
            "pixel_center_uv": self.pixel_center_uv,
            "visible_surface_point_color": array_or_none(
                self.visible_surface_point_color
            ),
            "estimated_object_center_color": array_or_none(
                self.estimated_object_center_color
            ),
        }


def load_localization_config(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _kernel(size):
    size = int(size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def segment_cube_rgb(color_rgb, config):
    """Return selected contour mask and diagnostics; no depth or truth input."""
    hsv_cfg = config["hsv_segmentation"]
    hsv = cv2.cvtColor(np.asarray(color_rgb), cv2.COLOR_RGB2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for bounds in hsv_cfg["ranges_opencv_hsv"]:
        combined |= cv2.inRange(
            hsv, np.array(bounds["lower"], np.uint8),
            np.array(bounds["upper"], np.uint8)
        )
    opened = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, _kernel(hsv_cfg["open_kernel"]),
        iterations=int(hsv_cfg["open_iterations"])
    )
    cleaned = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, _kernel(hsv_cfg["close_kernel"]),
        iterations=int(hsv_cfg["close_iterations"])
    )
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        box_area = float(w * h)
        aspect = float(w / h) if h else np.inf
        rectangularity = area / box_area if box_area else 0.0
        if not hsv_cfg["min_contour_area_px"] <= area <= hsv_cfg["max_contour_area_px"]:
            continue
        if not hsv_cfg["aspect_ratio_min"] <= aspect <= hsv_cfg["aspect_ratio_max"]:
            continue
        if rectangularity < hsv_cfg["min_rectangularity"]:
            continue
        # Prefer a large, filled, roughly square projected cube face.
        square_score = np.exp(-abs(np.log(max(aspect, 1e-9))))
        score = area * rectangularity * square_score
        candidates.append((score, contour, area, (x, y, w, h)))
    if not candidates:
        return None
    _, contour, area, bbox = max(candidates, key=lambda item: item[0])
    mask = np.zeros(cleaned.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-12:
        return None
    center = (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )
    eroded = cv2.erode(
        mask, _kernel(hsv_cfg["erode_kernel"]),
        iterations=int(hsv_cfg["erode_iterations"])
    )
    return {
        "mask": mask,
        "eroded_mask": eroded,
        "contour": contour,
        "contour_area_px": area,
        "bbox_xywh": bbox,
        "pixel_center_uv": center,
    }


def _failure(reason, segmentation, point_map=None, projected_count=0,
             mask_point_count=0):
    shape = (1, 1) if segmentation is None else segmentation["mask"].shape
    empty = np.zeros(shape, dtype=np.uint8)
    return CubeLocalizationResult(
        valid=False,
        failure_reason=reason,
        hsv_detected=segmentation is not None,
        contour_area_px=None if segmentation is None else segmentation["contour_area_px"],
        mask_pixel_count=0 if segmentation is None else int(np.count_nonzero(segmentation["mask"])),
        eroded_mask_pixel_count=0 if segmentation is None else int(np.count_nonzero(segmentation["eroded_mask"])),
        projected_depth_point_count=int(projected_count),
        mask_depth_point_count=int(mask_point_count),
        filtered_depth_point_count=0,
        valid_point_ratio=0.0,
        depth_median_m=None,
        depth_mad_m=None,
        bbox_xywh=None if segmentation is None else segmentation["bbox_xywh"],
        pixel_center_uv=None if segmentation is None else segmentation["pixel_center_uv"],
        visible_surface_point_color=None,
        estimated_object_center_color=None,
        mask=empty if segmentation is None else segmentation["mask"],
        eroded_mask=empty if segmentation is None else segmentation["eroded_mask"],
        projected_point_map=(
            np.full(shape + (3,), np.nan) if point_map is None else point_map
        ),
    )


def localize_cube_rgbd(color_rgb, depth, depth_intrinsics, color_intrinsics,
                       color_from_depth, support_up_color, config):
    """Localize the cube without simulator truth or robot-base coordinates."""
    segmentation = segment_cube_rgb(color_rgb, config)
    if segmentation is None:
        return _failure("no_valid_hsv_contour", None)

    point_map, projected_count = project_depth_points_to_color(
        depth, depth_intrinsics, color_intrinsics, color_from_depth
    )
    depth_cfg = config["depth_filter"]
    eroded = segmentation["eroded_mask"] > 0
    finite = np.isfinite(point_map).all(axis=2)
    legal = (
        finite & (point_map[..., 2] > depth_cfg["min_z_m"])
        & (point_map[..., 2] < depth_cfg["max_z_m"])
    )
    selected = point_map[eroded & legal]
    mask_point_count = int(selected.shape[0])
    if mask_point_count < int(depth_cfg["minimum_points"]):
        return _failure(
            "insufficient_projected_depth_points", segmentation, point_map,
            projected_count, mask_point_count
        )

    z = selected[:, 2]
    z_median = float(np.median(z))
    z_mad = float(np.median(np.abs(z - z_median)))
    robust_radius = max(
        float(depth_cfg["mad_scale"]) * z_mad,
        float(depth_cfg["minimum_mad_m"]),
    )
    low_q, high_q = np.quantile(
        z, [depth_cfg["lower_quantile"], depth_cfg["upper_quantile"]]
    )
    keep = (
        (np.abs(z - z_median) <= robust_radius)
        & (z >= low_q) & (z <= high_q)
    )
    filtered = selected[keep]
    if filtered.shape[0] < int(depth_cfg["minimum_points"]):
        return _failure(
            "insufficient_points_after_outlier_filter", segmentation, point_map,
            projected_count, mask_point_count
        )

    visible_surface = np.median(filtered, axis=0)
    support_up_color = np.asarray(support_up_color, dtype=np.float64)
    up_norm = float(np.linalg.norm(support_up_color))
    if not np.isfinite(up_norm) or up_norm < 1e-9:
        return _failure(
            "invalid_support_up_direction", segmentation, point_map,
            projected_count, mask_point_count
        )
    support_up_color /= up_norm
    # Initial explicit model: an upright 30 mm cube rests on a horizontal
    # support, and the downward-looking camera observes its top face. Move from
    # the robust visible top surface half a side opposite the support-up vector.
    estimated_center = visible_surface - (
        0.5 * float(config["object"]["side_length_m"]) * support_up_color
    )
    return CubeLocalizationResult(
        valid=True,
        failure_reason=None,
        hsv_detected=True,
        contour_area_px=segmentation["contour_area_px"],
        mask_pixel_count=int(np.count_nonzero(segmentation["mask"])),
        eroded_mask_pixel_count=int(np.count_nonzero(segmentation["eroded_mask"])),
        projected_depth_point_count=int(projected_count),
        mask_depth_point_count=mask_point_count,
        filtered_depth_point_count=int(filtered.shape[0]),
        valid_point_ratio=float(filtered.shape[0] / max(mask_point_count, 1)),
        depth_median_m=z_median,
        depth_mad_m=z_mad,
        bbox_xywh=segmentation["bbox_xywh"],
        pixel_center_uv=segmentation["pixel_center_uv"],
        visible_surface_point_color=visible_surface,
        estimated_object_center_color=estimated_center,
        mask=segmentation["mask"],
        eroded_mask=segmentation["eroded_mask"],
        projected_point_map=point_map,
    )
