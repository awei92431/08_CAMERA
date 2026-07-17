#!/usr/bin/env python3
"""Generate the exact ArUco texture referenced by scene_cube3cm.xml."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument(
        "--output", type=Path,
        default=root / "assets" / "aruco" / "aruco_4x4_50_id0.png")
    args = parser.parse_args()
    if not hasattr(cv2.aruco, args.dictionary):
        raise ValueError(f"unknown dictionary {args.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dictionary))
    marker = cv2.aruco.generateImageMarker(
        dictionary, args.marker_id, args.pixels)
    # RGB PNG avoids MuJoCo's single-channel texture restrictions.
    rgb = cv2.cvtColor(marker.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {args.output}")
    print(f"generated {args.dictionary} id={args.marker_id}: {args.output}")


if __name__ == "__main__":
    main()
