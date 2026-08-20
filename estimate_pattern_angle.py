#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# 기본 각도 스텝
DEFAULT_STEP = 0.1

# ------------------------------
# 유틸
# ------------------------------

def _resize_for_speed(img: np.ndarray, max_size: int) -> np.ndarray:
    """긴 변을 max_size로 리사이즈 (분석용), aspect 유지."""
    h, w = img.shape
    m = max(h, w)
    if m <= max_size:
        return img
    scale = max_size / float(m)
    return cv2.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)


def _apply_central_mask(img: np.ndarray, keep_ratio: float) -> np.ndarray:
    """원형 중앙부만 남김. keep_ratio는 지름 비율(0~1]. 1.0은 마스킹 없음."""
    if keep_ratio >= 0.999:
        return img
    h, w = img.shape
    cy, cx = h/2.0, w/2.0
    rad = 0.5 * keep_ratio * min(h, w)
    Y, X = np.ogrid[:h, :w]
    R = np.hypot(X - cx, Y - cy)
    mask = (R <= rad).astype(np.float32)
    return img * mask


# ------------------------------
# 수동 Radon (OpenCV 회전 → 컬럼 합) 
# ------------------------------

def compute_sinogram_manual(img: np.ndarray,
                            thetas: np.ndarray,
                            interpolation: int = cv2.INTER_LINEAR,
                            border_mode: int = cv2.BORDER_REFLECT101) -> np.ndarray:
    """
    이미지를 각 theta로 회전 → 컬럼 합(Projection) → sinogram 생성.
    반환 shape: (len(projection), len(thetas))
    """
    h, w = img.shape
    center = (w / 2.0, h / 2.0)
    sinogram = []

    for theta in thetas:
        M = cv2.getRotationMatrix2D(center, -theta, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), flags=interpolation, borderMode=border_mode)
        proj = rot.sum(axis=0)
        sinogram.append(proj)

    return np.array(sinogram).T


# ------------------------------
# 각도 추정 (Radon 기반, 속도 최적화 + 90° 감도 강화)
# ------------------------------

def estimate_rotation_radon(image_path: str,
                             step: float = DEFAULT_STEP,
                             debug: bool = True,
                             max_size: int = 768,
                             keep_ratio: float = 1.0,
                             refine_half: Optional[float] = None,
                             refine_step: Optional[float] = None):
    """
    수동 Radon 기반 회전각 추정 (빠름/안정성 개선):
      - 다운샘플(max_size)로 속도 ↑
      - 중앙 원형 마스크(keep_ratio)로 웨이퍼 둥근 에지 영향 ↓
      - thetas를 step/2에서 시작해 0/90° 스냅 회피
      - metric: s축 1차 차분 에너지(DC 억제) → 90° 부근 감도 유지
      - refine: coarse 근방만 고해상도 재탐색 + 3점 포물선 보간
    Returns (best_angle, coarse_theta, timings)
    """
    # 1) 로드 & 전처리 (윈도우 + 중앙 마스크 + 다운샘플)
    img0 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(f"Cannot read '{image_path}'")

    h0, w0 = img0.shape
    win0 = np.outer(np.hanning(h0), np.hanning(w0)).astype(np.float32)
    img_win = img0.astype(np.float32) * win0
    img_win = _apply_central_mask(img_win, keep_ratio)

    img_small = _resize_for_speed(img_win, max_size)
    h, w = img_small.shape

    # 2) 각도 그리드 (중간점 시작)
    thetas = np.arange(step/2, 180.0, step)

    # 타이머
    t0 = time.perf_counter()

    # 3) Sinogram (coarse: 빠른 보간/반사 경계)
    sinogram = compute_sinogram_manual(img_small, thetas, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT101)
    t1 = time.perf_counter()

    # 4) 고역 기반 지표
    s_mean = sinogram.mean(axis=0, keepdims=True)
    proj_hp = np.diff(sinogram - s_mean, axis=0)
    proj_metric = (proj_hp * proj_hp).mean(axis=0)

    coarse_idx = int(np.argmax(proj_metric))
    coarse_theta = float(thetas[coarse_idx])
    t2 = time.perf_counter()

    # 5) refine (작은 창만 정밀 보간)
    if refine_half is None:
        refine_half = max(0.5, 10 * step)
    if refine_step is None:
        refine_step = max(0.005, step / 5)  # 속도/정밀 균형

    ref_start = max(0.0, coarse_theta - refine_half)
    ref_end   = min(180.0 - refine_step, coarse_theta + refine_half)
    ref_thetas = np.arange(ref_start, ref_end + 1e-9, refine_step)

    def metric_for_angle(th: float) -> float:
        M = cv2.getRotationMatrix2D((w/2.0, h/2.0), -th, 1.0)
        rot = cv2.warpAffine(
            img_small, M, (w, h),
            flags=cv2.INTER_CUBIC,                # 정밀
            borderMode=cv2.BORDER_REFLECT101
        )
        proj = rot.sum(axis=0).astype(np.float32)
        proj -= proj.mean()
        hp = np.diff(proj)
        return float((hp * hp).mean())

    ref_metrics = np.array([metric_for_angle(th) for th in ref_thetas])
    j = int(np.argmax(ref_metrics))
    best_angle = float(ref_thetas[j])
    t3 = time.perf_counter()

    # 6) 3점 포물선 보간 (균일 간격)
    if 0 < j < len(ref_thetas) - 1:
        y1, y2, y3 = ref_metrics[j-1], ref_metrics[j], ref_metrics[j+1]
        hstep = ref_thetas[1] - ref_thetas[0]
        denom = (y1 - 2*y2 + y3)
        if abs(denom) > 1e-12:
            best_angle = ref_thetas[j] + 0.5 * (y1 - y3) / denom * hstep
    t4 = time.perf_counter()

    timings = {
        'sinogram_ms': (t1 - t0) * 1000.0,
        'coarse_ms'  : (t2 - t1) * 1000.0,
        'refine_ms'  : (t3 - t2) * 1000.0,
        'quad_ms'    : (t4 - t3) * 1000.0,
    }

    if debug:
        plt.figure(figsize=(6, 6 * (h/w)))
        plt.imshow(sinogram, cmap='gray', aspect='auto', extent=[0, 180, 0, sinogram.shape[0]])
        plt.title("Sinogram (OpenCV rotate, downsampled)")
        plt.xlabel("Projection Angle θ (°)"); plt.ylabel("Detector position s (px)")
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(thetas, proj_metric, '-', lw=1, label='Hi-pass energy')
        plt.axvline(coarse_theta, color='orange', linestyle='--', label=f'Coarse {coarse_theta:.4f}°')
        plt.title("Projection High-Frequency Energy vs Angle (coarse)")
        plt.xlabel("θ (°)"); plt.ylabel("Energy"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(ref_thetas, ref_metrics, '-', lw=1, label='Refined metric')
        plt.axvline(best_angle, color='r', linestyle='--', label=f'Best {best_angle:.5f}°')
        plt.title("Refined Metric in Local Window")
        plt.xlabel("θ (°)"); plt.ylabel("Energy"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return best_angle, coarse_theta, timings


# ------------------------------
# 시각화 저장
# ------------------------------

def annotate_and_save(image_path: str, angle_deg: float, out_path: str, arrow_mode: str = "line") -> None:
    """원본 이미지 위에 결과 각도(숫자만)와 화살표를 그려 저장."""
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        raise FileNotFoundError(f"Cannot read '{image_path}'")

    h, w = img_color.shape[:2]
    cx, cy = w // 2, h // 2

    draw_angle = (angle_deg + 90.0) % 180.0 if arrow_mode != 'normal' else (angle_deg % 180.0)

    rad = np.deg2rad(draw_angle)
    L = int(0.35 * min(h, w))
    x2 = int(cx + L * np.cos(rad))
    y2 = int(cy - L * np.sin(rad))

    cv2.arrowedLine(img_color, (cx, cy), (x2, y2), (0, 255, 0), 2, line_type=cv2.LINE_AA, tipLength=0.03)

    text = f"{angle_deg:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    pad = 12

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x0, y0 = pad, pad + th
    cv2.rectangle(img_color, (x0 - 8, y0 - th - 8), (x0 + tw + 8, y0 + baseline + 8), (0, 0, 0), -1)
    cv2.putText(img_color, text, (x0, y0), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if not cv2.imwrite(out_path, img_color):
        raise IOError(f"Failed to save annotated image to '{out_path}'")


# ------------------------------
# 메인 (파일/디렉터리 모두 지원)
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate image rotation via manual Radon transform and annotate result (file or directory).")
    parser.add_argument("path", help="Path to input image OR a directory containing images")
    parser.add_argument("-s", "--step", type=float, default=DEFAULT_STEP,
                        help="Angle step in degrees for coarse scan (default: 0.1°)")
    parser.add_argument("--max-size", type=int, default=768,
                        help="Downsample longer image side to this many pixels for speed (analysis only). Default: 768")
    parser.add_argument("--keep-ratio", type=float, default=1.0,
                        help="Keep only central circular region by this diameter ratio (0<r<=1). Default: 1.0 (no mask)")
    parser.add_argument("--refine-half", type=float, default=None,
                        help="Half window (deg) around coarse peak for refinement. Default: max(0.5, 10×step)")
    parser.add_argument("--refine-step", type=float, default=None,
                        help="Refine step (deg). Default: max(0.005, step/5)")
    parser.add_argument("--arrow", choices=["line", "normal"], default="line",
                        help="Arrow direction on output: 'line' (θ+90°) or 'normal' (θ). Default: line")
    parser.add_argument("-o", "--out", default=None,
                        help="Output image path for annotated result (file mode only). In directory mode, results are saved next to each input as <name>_annotated.png")
    parser.add_argument("--csv", default=None,
                        help="CSV output path. In directory mode default is <dir>/angles.csv; in file mode, if set, writes a single-row CSV.")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable plotting for batch runs (directory mode forces no-debug)")

    args = parser.parse_args()

    p = Path(args.path)
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    # 디렉터리 모드
    if p.is_dir():
        images = [str(f) for f in p.iterdir() if f.suffix.lower() in exts]
        if not images:
            print("No images found in directory.")
            return
        total = len(images)
        print(f"Found {total} images. Processing...")
        rows = []  # (filename, coarse_deg, best_deg)
        for i, img_path in enumerate(sorted(images), start=1):
            try:
                print(f"[{i}/{total}] Processing: {img_path}")
                best, coarse, t = estimate_rotation_radon(
                    img_path,
                    step=args.step,
                    debug=False,
                    max_size=args.max_size,
                    keep_ratio=args.keep_ratio,
                    refine_half=args.refine_half,
                    refine_step=args.refine_step,
                )
                out_path = str(Path(img_path).with_name(Path(img_path).stem + "_annotated.png"))
                annotate_and_save(img_path, best, out_path, arrow_mode=args.arrow)
                print(f"    Angles : coarse={coarse:.2f} deg  ->  best={best:.2f} deg")
                print(f"    Timings: sinogram {t['sinogram_ms']:.1f} ms | coarse {t['coarse_ms']:.1f} ms | refine {t['refine_ms']:.1f} ms | quad {t['quad_ms']:.1f} ms")
                print(f"    saved  : {out_path}")
                rows.append((Path(img_path).name, coarse, best))
            except Exception as e:
                print(f"    ERROR processing {img_path}: {e}")
        # CSV 작성
        csv_path = Path(args.csv) if args.csv else (p / "angles.csv")
        try:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                fieldnames = ["filename", "coarse_theta_deg", "best_angle_deg"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for name, coarse_v, best_v in rows:
                    writer.writerow({
                        "filename": name,
                        "coarse_theta_deg": f"{coarse_v:.2f}",
                        "best_angle_deg": f"{best_v:.2f}",
                    })
            print(f"CSV saved to: {csv_path}")
        except Exception as e:
            print(f"ERROR writing CSV to {csv_path}: {e}")
        return

    # 파일 모드
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if p.suffix.lower() not in exts:
        raise ValueError("Unsupported file extension. Supported: bmp, png, jpg, jpeg, tif, tiff")

    best, coarse, t = estimate_rotation_radon(
        str(p), step=args.step, debug=not args.no_debug,
        max_size=args.max_size, keep_ratio=args.keep_ratio,
        refine_half=args.refine_half, refine_step=args.refine_step
    )

    # 출력 경로 결정 (파일 모드)
    out_path = args.out or str(p.with_name(p.stem + "_annotated.png"))
    annotate_and_save(str(p), best, out_path, arrow_mode=args.arrow)

    print(f"Angles : coarse={coarse:.2f} deg  ->  best={best:.2f} deg")
    print(f"Timings: sinogram {t['sinogram_ms']:.1f} ms | coarse {t['coarse_ms']:.1f} ms | refine {t['refine_ms']:.1f} ms | quad {t['quad_ms']:.1f} ms")
    print(f"Annotated result saved to: {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        try:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                fieldnames = ["filename", "coarse_theta_deg", "best_angle_deg"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "filename": p.name,
                    "coarse_theta_deg": f"{coarse:.2f}",
                    "best_angle_deg": f"{best:.2f}",
                })
            print(f"CSV saved to: {csv_path}")
        except Exception as e:
            print(f"ERROR writing CSV to {csv_path}: {e}")


if __name__ == "__main__":
    main()
