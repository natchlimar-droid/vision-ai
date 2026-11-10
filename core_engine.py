"""
core_engine.py - Decoupled Headless Industrial Vision Inspection Engine.
Runs continuous real-time acquisition, OpenCV/Contour skew measurement,
OCR date validation, pharmaceutical sanity verification, and hardware I/O actuation.
"""

import time
import os
import threading
import cv2
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from queue import Queue

from inspector import IndustrialInspector
import database as db
from video_stream import FreshFrameStreamer
from hardware_io import IndustrialActuatorController


class CoreInspectionService:
    """
    Decoupled Headless Inspection Service.
    Operates independently of any UI frontend (Streamlit, FastAPI, PyQt, or WebSocket).
    Runs real-time inspection loop, records GMP logs to SQLite WAL, and dispatches events.
    """

    def __init__(
        self,
        camera_src: str = "0",
        skew_threshold: float = 3.0,
        batch_id: str = "BATCH-2026-01",
        on_inspection_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.camera_src = camera_src
        self.skew_threshold = skew_threshold
        self.batch_id = batch_id
        self.on_inspection_callback = on_inspection_callback

        self.inspector = IndustrialInspector(skew_threshold=skew_threshold)
        self.actuator = IndustrialActuatorController()
        self.streamer: Optional[FreshFrameStreamer] = None

        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.latest_result: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()
        self.event_queue: Queue = Queue(maxsize=100)

        # Initialize SQLite database with WAL mode
        db.init_db()

    def start(self) -> bool:
        """Starts video capture and background inspection worker."""
        if self.is_running:
            return True

        self.streamer = FreshFrameStreamer(self.camera_src)
        if not self.streamer.start():
            return False

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        return True

    def _worker_loop(self):
        """Continuous inspection loop running in background at max frame rate."""
        while self.is_running:
            if not self.streamer:
                time.sleep(0.05)
                continue

            ret, frame = self.streamer.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            # Run full inspection
            result = self.inspect_frame(frame)

            with self.lock:
                self.latest_result = result

            # Push to queue (drop oldest if full)
            if self.event_queue.full():
                try:
                    self.event_queue.get_nowait()
                except Exception:
                    pass
            self.event_queue.put(result)

            if self.on_inspection_callback:
                try:
                    self.on_inspection_callback(result)
                except Exception:
                    pass

            # Frame rate throttle (e.g. 15-20 FPS)
            time.sleep(0.05)

    def inspect_frame(self, frame: np.ndarray, roi_bbox: Optional[tuple] = None) -> Dict[str, Any]:
        """Runs single-frame QA, triggers hardware actuators, and logs to SQLite WAL."""
        res = self.inspector.inspect_frame(frame, roi_bbox=roi_bbox)
        s_inf = res.get("sachet_info", {})

        # Hardware actuation (Andon Tower & Solenoid Ejector)
        act_res = self.actuator.trigger_verdict_actuation(res["verdict"], res)
        res["actuation"] = act_res

        # Thread-safe write to SQLite in WAL mode
        log_id = db.log_inspection(
            skew_angle=res["skew_angle"],
            angle_status=res["angle_status"],
            date_found=res["date_found"],
            ocr_status=res["ocr_status"],
            verdict=res["verdict"],
            raw_text=res["raw_text"],
            cycle_time_ms=res["cycle_time_ms"],
            drug_name=s_inf.get("drug_name", "-"),
            dosage=s_inf.get("dosage", "-"),
            lot_no=s_inf.get("lot_no", "-"),
            mfg_date=s_inf.get("mfg_date", "-"),
            reg_no=s_inf.get("reg_no", "-"),
            batch_id=self.batch_id,
            sanity_status=res.get("sanity_status", "PASS"),
            defect_type=res.get("defect_type", "NONE"),
            master_match=res.get("master_match", "MATCHED")
        )
        res["log_id"] = log_id
        return res

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent inspection result."""
        with self.lock:
            return self.latest_result

    def stop(self):
        """Stops the inspection service and releases camera resources."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if self.streamer:
            self.streamer.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Industrial Vision QA - Decoupled Headless Core Engine")
    parser.add_argument("--src", type=str, default="samples/sachet_pass.jpg", help="Camera URL (RTSP/HTTP) or Image File")
    args = parser.parse_args()

    print(f"=== Core Inspection Engine Initialized (Target: {args.src}) ===")
    inspector = IndustrialInspector()

    if os.path.exists(args.src):
        img = cv2.imread(args.src)
        result = inspector.inspect_frame(img)
        print(f"Verdict        : {result['verdict']}")
        print(f"Skew Angle     : {result['skew_angle']} deg ({result['angle_status']})")
        print(f"Date Detected  : {result['date_found']} ({result['ocr_status']})")
        print(f"Sanity Status  : {result['sanity_status']} (Defect: {result['defect_type']})")
        print(f"Shelf Life     : {result.get('calculated_shelf_life')} months")
        print(f"Sachet Info    : {result['sachet_info']}")
        print(f"Cycle Time     : {result['cycle_time_ms']} ms")
    else:
        print(f"Starting background streaming worker on: {args.src}")
        service = CoreInspectionService(camera_src=args.src)
        service.start()
        try:
            while True:
                time.sleep(1.0)
                res = service.get_latest_result()
                if res:
                    print(f"[{res['verdict']}] Skew: {res['skew_angle']} deg | EXP: {res['date_found']} | Sanity: {res['sanity_status']}")
        except KeyboardInterrupt:
            service.stop()
            print("Engine stopped.")
