# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Real-Time Multi-Camera Streaming & Buffer Manager
-------------------------------------------------
Provides high-performance asynchronous multi-camera capture with ring buffers
for OpenCV (V4L2), Intel RealSense, and Mock streams.
"""

from abc import ABC, abstractmethod
from collections import deque
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("GR00T_CameraManager")


class BaseCameraStreamer(ABC):
    """Abstract base class for asynchronous camera streamers."""

    def __init__(self, camera_id: Union[int, str], width: int = 320, height: int = 240, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0

    @abstractmethod
    def start(self):
        """Starts asynchronous camera capture thread."""
        pass

    @abstractmethod
    def stop(self):
        """Stops camera capture and releases hardware device."""
        pass

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns the most recent RGB uint8 frame (H, W, 3)."""
        with self._lock:
            if self._latest_frame is not None:
                return np.array(self._latest_frame, copy=True)
            return None


class OpenCVCameraStreamer(BaseCameraStreamer):
    """Real-time OpenCV V4L2 USB camera streamer running in background thread."""

    def __init__(self, camera_id: Union[int, str] = 0, width: int = 320, height: int = 240, fps: int = 30):
        super().__init__(camera_id, width, height, fps)
        self._cap = None

    def start(self):
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for OpenCVCameraStreamer. Install via: pip install opencv-python")

        if self.is_running:
            return

        # Open video device
        dev_id = int(self.camera_id) if str(self.camera_id).isdigit() else self.camera_id
        self._cap = cv2.VideoCapture(dev_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open OpenCV camera device: {self.camera_id}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[OpenCV Camera] Streamer started on device: {self.camera_id} ({self.width}x{self.height} @ {self.fps} FPS)")

    def _capture_loop(self):
        import cv2
        while self.is_running and self._cap.isOpened():
            ret, frame_bgr = self._cap.read()
            if ret and frame_bgr is not None:
                # Convert BGR to RGB and resize if necessary
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[0] != self.height or frame_rgb.shape[1] != self.width:
                    frame_rgb = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)

                with self._lock:
                    self._latest_frame = frame_rgb
                    self._frame_count += 1
            else:
                time.sleep(0.005)

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info(f"[OpenCV Camera] Streamer stopped on device: {self.camera_id}")


class RealSenseCameraStreamer(BaseCameraStreamer):
    """Intel RealSense RGB-D camera streamer."""

    def __init__(self, camera_id: str = "", width: int = 640, height: int = 480, fps: int = 30):
        super().__init__(camera_id, width, height, fps)
        self._pipeline = None

    def start(self):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError("pyrealsense2 is required for RealSenseCameraStreamer. Install via: pip install pyrealsense2")

        if self.is_running:
            return

        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.camera_id:
            config.enable_device(str(self.camera_id))
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

        self._pipeline.start(config)
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[RealSense Camera] Streamer started on serial: {self.camera_id or 'Default'}")

    def _capture_loop(self):
        import cv2
        while self.is_running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame_rgb = np.asanyarray(color_frame.get_data())
                with self._lock:
                    self._latest_frame = frame_rgb
                    self._frame_count += 1
            except Exception as e:
                logger.warning(f"[RealSense Camera] Frame drop: {e}")
                time.sleep(0.01)

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._pipeline:
            self._pipeline.stop()
        logger.info(f"[RealSense Camera] Streamer stopped.")


class MockCameraStreamer(BaseCameraStreamer):
    """Dummy camera streamer for hardware dry-runs and offline latency tests."""

    def __init__(self, camera_id: str = "mock", width: int = 320, height: int = 240, fps: int = 30):
        super().__init__(camera_id, width, height, fps)

    def start(self):
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        interval = 1.0 / self.fps
        while self.is_running:
            # Generate dummy frame with timestamp pattern
            dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            dummy[:, :, 0] = (self._frame_count * 2) % 255
            dummy[:, :, 1] = (self._frame_count * 3) % 255
            dummy[:, :, 2] = (self._frame_count * 5) % 255

            with self._lock:
                self._latest_frame = dummy
                self._frame_count += 1

            time.sleep(interval)

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)


class MultiCameraManager:
    """
    Manages multiple active camera feeds and maintains synchronized ring buffers
    for GR00T policy observation video inputs across temporal horizon T.
    """

    def __init__(
        self,
        camera_map: Dict[str, BaseCameraStreamer],
        buffer_size: int = 5,
    ):
        """
        Args:
            camera_map: Dictionary mapping policy video key (e.g. 'ego_view_res320x240_freq20')
                        to camera streamer instance.
            buffer_size: Number of past frames to keep in memory per camera.
        """
        self.camera_map = camera_map
        self.buffer_size = buffer_size
        self.buffers: Dict[str, deque] = {
            key: deque(maxlen=buffer_size) for key in camera_map.keys()
        }

    def start_all(self):
        """Starts all camera streaming background threads."""
        for key, streamer in self.camera_map.items():
            print(f"[CameraManager] Starting camera stream for key: '{key}' (Device: {streamer.camera_id})")
            streamer.start()

    def stop_all(self):
        """Stops all camera streaming threads."""
        for key, streamer in self.camera_map.items():
            streamer.stop()

    def update_buffers(self):
        """Polls latest frames from streamers and appends to ring buffers."""
        for key, streamer in self.camera_map.items():
            frame = streamer.get_latest_frame()
            if frame is not None:
                self.buffers[key].append(frame)

    def get_observation_video(self, video_keys: List[str], horizon_t: int = 1) -> Dict[str, np.ndarray]:
        """
        Formats camera ring buffer into policy input video dict of shape (B=1, T, H, W, 3).

        Args:
            video_keys: List of required video observation keys
            horizon_t: Temporal horizon length (T)

        Returns:
            Dictionary mapping video key -> numpy array (1, T, H, W, 3)
        """
        self.update_buffers()
        video_obs = {}

        for key in video_keys:
            buf = self.buffers.get(key, [])
            if len(buf) == 0:
                # If buffer is still empty, query streamer directly or fallback to black frame
                streamer = self.camera_map.get(key)
                frame = streamer.get_latest_frame() if streamer else None
                if frame is None:
                    h, w = (240, 320) if "320x240" in key else (224, 224)
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                buf_frames = [frame] * horizon_t
            elif len(buf) < horizon_t:
                # Replicate oldest frame to satisfy horizon_t
                oldest = buf[0]
                buf_frames = [oldest] * (horizon_t - len(buf)) + list(buf)
            else:
                # Take latest horizon_t frames
                buf_frames = list(buf)[-horizon_t:]

            # Stack into (T, H, W, 3) then expand batch dim to (1, T, H, W, 3)
            seq = np.stack(buf_frames, axis=0)
            video_obs[key] = np.expand_dims(seq, axis=0)

        return video_obs
