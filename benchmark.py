"""
benchmark.py - Automated Batch Testing & Accuracy Evaluation Harness.
Runs the Industrial Inspection Engine over a folder of test images,
measuring Skew accuracy, OCR detection rate, and Cycle Time (ms).
"""

import os
import time
import cv2
import pandas as pd
from tabulate import tabulate
from inspector import IndustrialInspector
import generate_sample

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")


def run_benchmark():
    print("=" * 65)
    print(" INDUSTRIAL QUALITY INSPECTION - BENCHMARK & ACCURACY SUITE ")
    print("=" * 65)

    # Ensure synthetic test samples exist
    generate_sample.generate_all()

    inspector = IndustrialInspector(
        skew_threshold=3.0,
        enable_dot_matrix_filter=True,
        dilation_kernel_size=2
    )

    image_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"[!] No test images found in {SAMPLE_DIR}")
        return

    records = []

    for img_name in sorted(image_files):
        img_path = os.path.join(SAMPLE_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        start_time = time.perf_counter()
        result = inspector.inspect_frame(img)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        records.append({
            "File Name": img_name,
            "Skew (deg)": f"{result['skew_angle']:+.2f}°",
            "Skew Status": result['angle_status'],
            "Date Detected": result['date_found'] or "NONE",
            "Sanity": result.get('sanity_status', 'PASS'),
            "Defect": result.get('defect_type', 'NONE'),
            "Verdict": result['verdict'],
            "Cycle Time (ms)": f"{elapsed_ms:.1f} ms"
        })

    df = pd.DataFrame(records)

    print("\n[RESULT TABLE]")
    try:
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    except ImportError:
        print(df.to_string(index=False))

    total = len(records)
    passes = sum(1 for r in records if r["Verdict"] == "PASS")
    rejects = total - passes
    avg_latency = df["Cycle Time (ms)"].apply(lambda x: float(x.replace(" ms", ""))).mean()

    print("\n" + "=" * 65)
    print(f" Total Tests     : {total}")
    print(f" Total PASS      : {passes}")
    print(f" Total REJECT    : {rejects}")
    print(f" Avg Cycle Time  : {avg_latency:.1f} ms")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()

