#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a sequence of rotated images around a specified rotation center.

Usage examples (Windows PowerShell):
  # 0,10,20,...,170 deg around image center; save PNGs + angles.csv manifest
  python gen_rotate_image.py "C:\\data\\wafer.png" "C:\\data\\rot_out"

  # Custom center in pixels (x,y), custom angle range start:end:step (inclusive)
  python gen_rotate_image.py img.bmp out_dir --center 1024,980 --angles 0:170:10

  # Center as relative fractions of width/height (0..1), draw a marker
  python gen_rotate_image.py img.bmp out_dir --center 0.52,0.48 --mark-center

  # Provide explicit angle list (comma-separated, can be negatives/decimals)
  python gen_rotate_image.py img.bmp out_dir --angles-list "-40,-20,0,20,40"

  # Expand canvas to fit the whole rotated content instead of keeping size
  python gen_rotate_image.py img.bmp out_dir --expand-canvas

Notes
-----
- Writes images named like: <stem>_rot_<sign><angle with 2 decimals>.png
- Also writes a CSV manifest: angles.csv with columns: filename,angle_deg,center_x,center_y
- Works for grayscale or color images.
"""

import sys
import csv
import math
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ------------------------------
# Helpers
# ------------------------------

def parse_center(center_str: str, w: int, h: int) -> Tuple[float, float]:
    """Parse center as 'x,y'. If both <= 1.0, treat as relative fractions."""
    if center_str.lower() == 'auto':
        return (w / 2.0, h / 2.0)
    parts = center_str.split(',')
    if len(parts) != 2:
        raise ValueError("--center must be 'x,y' or 'auto'")
    x = float(parts[0].strip())
    y = float(parts[1].strip())
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return (x * w, y * h)
    return (x, y)


def parse_angles_spec(spec: str) -> List[float]:
    """Parse 'start:end:step' (inclusive end if aligned)."""
    parts = spec.split(':')
    if len(parts) != 3:
        raise ValueError("--angles must be 'start:end:step'")
    start = float(parts[0]); end = float(parts[1]); step = float(parts[2])
    if step == 0:
        raise ValueError("step cannot be 0")
    # Use np.arange with tiny epsilon to include end when aligned
    import numpy as _np
    arr = _np.arange(start, end + math.copysign(1e-9, step), step)
    return [float(a) for a in arr.tolist()]


def sanitize_angle_token(angle: float) -> str:
    """Turn angle into a filename-friendly token, e.g., -40.00 -> m40.00, +5.50 -> p05.50"""
    s = f"{angle:+06.2f}"  # e.g., +00.00, -40.00
    s = s.replace('+', 'p').replace('-', 'm')
    return s


def compute_expanded_size(w: int, h: int, M: np.ndarray) -> Tuple[int, int, float, float]:
    """Given 2x3 M, compute bounding box size and required translation (tx, ty) to keep all pixels.
    Returns (new_w, new_h, add_tx, add_ty) where you should do:
        M2 = M.copy(); M2[0,2] += add_tx; M2[1,2] += add_ty
    """
    corners = np.array([[0,0,1], [w,0,1], [0,h,1], [w,h,1]], dtype=np.float32).T  # 3x4
    trans = (M @ corners).T  # 4x2
    xs = trans[:,0]; ys = trans[:,1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    new_w = int(math.ceil(maxx - minx))
    new_h = int(math.ceil(maxy - miny))
    add_tx = -minx
    add_ty = -miny
    return new_w, new_h, add_tx, add_ty


# ------------------------------
# Core
# ------------------------------

def generate_rotations(in_path: Path,
                       out_dir: Path,
                       angles: List[float],
                       center_xy: Tuple[float, float],
                       keep_size: bool = True,
                       mark_center: bool = False,
                       border_mode: str = 'constant',
                       border_value: int = 0,
                       out_ext: str = '.png') -> None:
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {in_path}")
    h, w = img.shape[:2]
    cx, cy = center_xy

    border_map = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,
        'reflect101': cv2.BORDER_REFLECT_101,
        'replicate': cv2.BORDER_REPLICATE,
        'wrap': cv2.BORDER_WRAP,
    }
    if border_mode not in border_map:
        raise ValueError("Invalid --border-mode. Choose from constant, reflect, reflect101, replicate, wrap")
    bmode = border_map[border_mode]

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for ang in angles:
        # Rotation around (cx, cy)
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)

        if keep_size:
            dst_size = (w, h)
            M2 = M
        else:
            new_w, new_h, add_tx, add_ty = compute_expanded_size(w, h, M)
            M2 = M.copy()
            M2[0,2] += add_tx
            M2[1,2] += add_ty
            dst_size = (new_w, new_h)

        rotated = cv2.warpAffine(
            img, M2, dst_size,
            flags=cv2.INTER_CUBIC,
            borderMode=bmode,
            borderValue=border_value
        )

        if mark_center:
            # Draw a small cross at the rotation center location in the rotated image
            cpt = np.array([[cx, cy, 1.0]], dtype=np.float32).T  # 3x1
            if keep_size:
                c2 = (M2 @ cpt).ravel()
            else:
                c2 = (M2 @ cpt).ravel()
            x2, y2 = int(round(c2[0])), int(round(c2[1]))
            cv2.drawMarker(rotated, (x2, y2), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

        token = sanitize_angle_token(ang)
        out_name = f"{in_path.stem}_rot_{token}{out_ext}"
        out_path = out_dir / out_name
        ok = cv2.imwrite(str(out_path), rotated)
        if not ok:
            raise IOError(f"Failed to save {out_path}")

        manifest_rows.append([out_name, f"{ang:.2f}", f"{cx:.2f}", f"{cy:.2f}"])
        print(f"Saved: {out_path} (angle={ang:.2f}°, center=({cx:.2f},{cy:.2f}))")

    # manifest CSV
    csv_path = out_dir / 'angles.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as fh:
        wri = csv.writer(fh)
        wri.writerow(['filename','angle_deg','center_x','center_y'])
        wri.writerows(manifest_rows)
    print(f"Manifest saved to: {csv_path}")


# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description='Generate rotated images around a specified center for testing.')
    ap.add_argument('image', help='Input image path')
    ap.add_argument('outdir', help='Output directory to save rotated images')
    ap.add_argument('--center', default='auto', help="Rotation center as 'x,y' in pixels or fractions (0..1,0..1). Default: auto=image center")
    ap.add_argument('--angles', default='0:170:10', help="Angle spec 'start:end:step' in degrees (inclusive when aligned). Default: 0:170:10")
    ap.add_argument('--angles-list', default=None, help='Comma-separated explicit angle list (e.g., "-40,-20,0,20,40"). Overrides --angles if set')
    ap.add_argument('--keep-size', action='store_true', help='Keep original canvas size (default)')
    ap.add_argument('--expand-canvas', action='store_true', help='Expand output canvas to fit full rotated content')
    ap.add_argument('--mark-center', action='store_true', help='Draw a green cross at the rotation center in each output')
    ap.add_argument('--border-mode', default='constant', choices=['constant','reflect','reflect101','replicate','wrap'], help='Border mode for warpAffine (default: constant)')
    ap.add_argument('--border-value', type=int, default=0, help='Border value (grayscale or applied to all channels) if border-mode=constant (default: 0)')
    ap.add_argument('--ext', default='.png', help='Output image extension (e.g., .png, .bmp, .jpg). Default: .png')
    args = ap.parse_args()

    in_path = Path(args.image)
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        sys.exit(1)

    out_dir = Path(args.outdir)

    # read once to know size
    img0 = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img0 is None:
        print(f"Cannot read input: {in_path}")
        sys.exit(1)
    h, w = img0.shape[:2]

    try:
        cx, cy = parse_center(args.center, w, h)
    except Exception as e:
        print(f"Failed to parse --center: {e}")
        sys.exit(1)

    try:
        if args.angles_list:
            angles = [float(a.strip()) for a in args.angles_list.split(',') if a.strip()]
        else:
            angles = parse_angles_spec(args.angles)
    except Exception as e:
        print(f"Failed to parse angles: {e}")
        sys.exit(1)

    if args.expand_canvas and args.keep_size:
        print("Both --keep-size and --expand-canvas set. Using --expand-canvas.")

    keep_size = not args.expand_canvas

    generate_rotations(
        in_path,
        out_dir,
        angles,
        (cx, cy),
        keep_size=keep_size,
        mark_center=args.mark_center,
        border_mode=args.border_mode,
        border_value=args.border_value,
        out_ext=args.ext,
    )


if __name__ == '__main__':
    main()
