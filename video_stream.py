"""
video_stream.py - High-performance, zero-latency video stream handler.
Solves buffer bloat and lag from IP cameras / smartphone video streams using 
a dedicated background daemon thread.
"""

import cv2
import threading
import time
import requests
import numpy as np
from typing import Tuple, Optional


class FreshFrameStreamer:
    """
    Threaded video capture that continuously consumes frames from an RTSP / HTTP
    stream so that the caller always receives the freshest frame with zero lag.
    Uses pure-Python HTTP streaming to prevent FFmpeg libavformat C-assertion crashes
    on unstable network / smartphone streams.
    """

    def __init__(self, src: str, timeout: float = 5.0):
        self.src = src.strip()
        self.timeout = timeout
        self.latest_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_http = self.src.startswith("http://") or self.src.startswith("https://")
        self.is_snapshot_url = self.src.endswith(".jpg") or "shot.jpg" in self.src

    def start(self) -> bool:
        """Starts the capture background thread."""
        if self.is_running:
            return True

        if not self.is_http:
            # Handle integer webcam index (e.g. "0") or RTSP
            if self.src.isdigit():
                self.cap = cv2.VideoCapture(int(self.src))
            else:
                self.cap = cv2.VideoCapture(self.src)

            if not self.cap or not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # Wait up to 3.5 seconds for initial frame
        start_wait = time.time()
        while time.time() - start_wait < 3.5:
            if self.latest_frame is not None:
                return True
            time.sleep(0.05)

        return self.latest_frame is not None

    def _capture_loop(self):
        """Continuously grabs frames in background with zero-buffer and auto-reconnect."""
        while self.is_running:
            try:
                if self.is_http:
                    # Pure-Python HTTP MJPEG / Snapshot Streamer
                    # Eliminates OpenCV/FFmpeg C-assertion crashes when phone locks or Wi-Fi stutters
                    if self.is_snapshot_url:
                        resp = requests.get(self.src, timeout=self.timeout)
                        if resp.status_code == 200:
                            img_arr = np.frombuffer(resp.content, dtype=np.uint8)
                            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self.lock:
                                    self.latest_frame = frame
                        time.sleep(0.03)
                    else:
                        # Stream multipart/x-mixed-replace MJPEG via pure Python socket
                        with requests.get(self.src, stream=True, timeout=self.timeout) as r:
                            if r.status_code != 200:
                                time.sleep(0.5)
                                continue
                            bytes_buf = b""
                            for chunk in r.iter_content(chunk_size=16384):
                                if not self.is_running:
                                    break
                                bytes_buf += chunk
                                a = bytes_buf.find(b'\xff\xd8')
                                b = bytes_buf.find(b'\xff\xd9')
                                if a != -1 and b != -1:
                                    jpg = bytes_buf[a:b+2]
                                    bytes_buf = bytes_buf[b+2:]
                                    arr = np.frombuffer(jpg, dtype=np.uint8)
                                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        with self.lock:
                                            self.latest_frame = frame
                else:
                    if self.cap is None or not self.cap.isOpened():
                        time.sleep(0.1)
                        continue

                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                    else:
                        time.sleep(0.02)
            except Exception:
                time.sleep(0.5)  # Auto-reconnect backoff on network drop

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the freshest frame without buffering delays."""
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        return False, None

    def stop(self):
        """Stops the worker thread and releases OpenCV VideoCapture."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.latest_frame = None

