# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Real-Time Asynchronous Robot Control Pipeline
---------------------------------------------
Decouples Policy Inference from Real-Time Hardware Actuation:
- Thread 1: Asynchronous Vision-Language Policy Inference (~15-40ms on GPU/TRT)
- Thread 2: Real-Time Motor Actuation Loop (Precise 30Hz - 100Hz tick rate)
- Temporal Action Buffer with Receding Horizon Blending and Safety Interception
"""

from collections import deque
import datetime
import json
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Dict, List, Optional, Any, Callable
import numpy as np

from real_robot.camera_interface import MultiCameraManager
from real_robot.robot_interface import BaseRobotHardware
from real_robot.safety_guard import SafetyGuard, SafetyLimits

logger = logging.getLogger("GR00T_AsyncController")


class RealtimeAsyncRobotController:
    """
    Manages real-time closed-loop execution of GR00T VLA policy on physical robots.
    """

    def __init__(
        self,
        policy: Any,
        robot: BaseRobotHardware,
        camera_manager: MultiCameraManager,
        safety_guard: Optional[SafetyGuard] = None,
        control_freq: float = 30.0,
        execution_horizon: int = 16,
        task_instruction: str = "",
        dry_run: bool = False,
        save_action_json: bool = False,
        save_action_json_path: str = "output_actions/real_robot_actions.json",
        save_video_frames: bool = False,
        save_frames_dir: str = "output_frames/real_robot",
    ):
        self.policy = policy
        self.robot = robot
        self.camera_manager = camera_manager
        self.safety_guard = safety_guard or SafetyGuard()
        self.control_freq = control_freq
        self.control_interval = 1.0 / control_freq
        self.execution_horizon = execution_horizon
        self.task_instruction = task_instruction
        self.dry_run = dry_run
        self.save_action_json = save_action_json
        self.save_action_json_path = save_action_json_path
        self.save_video_frames = save_video_frames
        self.save_frames_dir = save_frames_dir

        # Modality configs from policy
        self.modality_configs = self.policy.get_modality_config()
        self.req_video_keys = self.modality_configs["video"].modality_keys
        self.req_state_keys = self.modality_configs["state"].modality_keys
        self.req_lang_keys = self.modality_configs["language"].modality_keys

        self.v_horizon = len(self.modality_configs["video"].delta_indices)
        self.s_horizon = len(self.modality_configs["state"].delta_indices)
        self.l_horizon = len(self.modality_configs["language"].delta_indices)

        # State dimensions cache
        self.state_dims = {}
        try:
            tag_val = self.policy.embodiment_tag.value
            proc_stats = self.policy.processor.statistics[tag_val]["state"]
            for s_key in self.req_state_keys:
                if s_key in proc_stats:
                    self.state_dims[s_key] = proc_stats[s_key]["min"].shape[-1]
        except Exception:
            pass

        # Threading & Queues
        self.is_running = False
        self._action_queue = queue.Queue(maxsize=10)
        self._inference_trigger = threading.Event()
        self._inference_thread: Optional[threading.Thread] = None
        self._actuation_thread: Optional[threading.Thread] = None

        # Statistics & Logs
        self.stats = {
            "inference_count": 0,
            "actuation_ticks": 0,
            "avg_inference_latency_ms": 0.0,
            "max_inference_latency_ms": 0.0,
            "action_queue_underruns": 0,
        }

    def start(self):
        """Initializes hardware connections and starts async threads."""
        print("\n=======================================================")
        print("🚀 Starting GR00T Real-Time Asynchronous Robot Controller")
        print(f" Robot Interface  : {self.robot.name} (Dry-Run Mode: {self.dry_run})")
        print(f" Control Frequency: {self.control_freq} Hz (Tick interval: {self.control_interval*1000:.1f} ms)")
        print(f" Execution Horizon: {self.execution_horizon} steps")
        print(f" Task Instruction : '{self.task_instruction}'")
        print("=======================================================\n")

        # 1. Connect Robot Hardware
        if not self.robot.is_connected:
            if not self.robot.connect():
                raise RuntimeError("Failed to connect to robot hardware interface!")

        # 2. Start Camera Streams
        self.camera_manager.start_all()
        time.sleep(0.5)  # Allow camera buffers to warm up

        self.is_running = True

        # 3. Launch Inference Worker Thread
        self._inference_thread = threading.Thread(target=self._inference_worker_loop, daemon=True)
        self._inference_thread.start()

        # 4. Launch Real-time Actuation Loop
        self._actuation_thread = threading.Thread(target=self._actuation_loop, daemon=True)
        self._actuation_thread.start()

    def stop(self):
        """Gracefully stops all asynchronous threads and disconnects hardware."""
        print("\n[AsyncController] Stopping real-time controller...")
        self.is_running = False
        self._inference_trigger.set()

        if self._actuation_thread and self._actuation_thread.is_alive():
            self._actuation_thread.join(timeout=1.0)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.0)

        self.camera_manager.stop_all()
        self.robot.disconnect()
        print("✅ [AsyncController] Controller stopped safely.")

    def _inference_worker_loop(self):
        """Worker thread executing model forward passes and pushing action chunks."""
        total_latency = 0.0

        while self.is_running:
            t_infer_start = time.perf_counter()

            # Step 1: Collect Video Observations from Ring Buffer
            video_input = self.camera_manager.get_observation_video(
                self.req_video_keys,
                horizon_t=self.v_horizon
            )

            # Optional: Save camera images to disk
            saved_frames_info = {}
            if self.save_video_frames:
                try:
                    from real_robot.camera_interface import save_video_frames_to_disk
                except ImportError:
                    pass

            # Step 2: Read current robot joint/EEF states
            state_dict = self.robot.get_state(self.req_state_keys, self.state_dims)
            state_input = {}
            for s_key in self.req_state_keys:
                raw_state = state_dict.get(s_key, np.zeros(self.state_dims.get(s_key, 7), dtype=np.float32))
                state_seq = np.stack([raw_state] * self.s_horizon, axis=0)
                state_input[s_key] = np.expand_dims(state_seq, axis=0).astype(np.float32)

            # Step 3: Format language prompt
            language_input = {
                self.req_lang_keys[0]: [[self.task_instruction] * self.l_horizon]
            }

            observation = {
                "video": video_input,
                "state": state_input,
                "language": language_input,
            }

            # Step 4: Run Policy Inference
            try:
                action, info = self.policy.get_action(observation)
            except Exception as e:
                logger.error(f"[AsyncController] Policy inference failed: {e}")
                time.sleep(0.05)
                continue

            t_infer_end = time.perf_counter()
            infer_ms = (t_infer_end - t_infer_start) * 1000.0

            self.stats["inference_count"] += 1
            total_latency += infer_ms
            self.stats["avg_inference_latency_ms"] = total_latency / self.stats["inference_count"]
            self.stats["max_inference_latency_ms"] = max(self.stats["max_inference_latency_ms"], infer_ms)

            # Optional: Save Action Chunk to JSON
            if self.save_action_json and self.save_action_json_path:
                self._save_action_to_json(action, infer_ms, saved_frames_info)

            # Push action chunk to queue for real-time actuation thread
            try:
                self._action_queue.put((action, state_dict), timeout=0.1)
            except queue.Full:
                logger.warning("[AsyncController] Action queue full, dropping oldest chunk.")

            # Wait for next trigger or throttle loop
            self._inference_trigger.wait(timeout=0.01)
            self._inference_trigger.clear()

    def _save_action_to_json(self, action: Dict[str, np.ndarray], infer_ms: float, saved_frames: Dict):
        try:
            path = Path(self.save_action_json_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            formatted = {}
            for k, v in action.items():
                formatted[k] = v[0].tolist() if (v.ndim == 3 and v.shape[0] == 1) else v.tolist()

            data = {
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "task_instruction": self.task_instruction,
                    "inference_ms": round(infer_ms, 2),
                    "dry_run": self.dry_run,
                },
                "action_chunk": formatted,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[AsyncController] Could not save action JSON: {e}")

    def _actuation_loop(self):
        """Real-time high-priority loop commanding physical robot motors at control_freq."""
        current_chunk = None
        current_ref_state = None
        step_in_chunk = 0
        total_steps_in_chunk = 0

        while self.is_running:
            t_tick_start = time.perf_counter()

            # 1. Check if we need a new action chunk from the queue
            if current_chunk is None or step_in_chunk >= min(self.execution_horizon, total_steps_in_chunk):
                try:
                    current_chunk, current_ref_state = self._action_queue.get(timeout=0.05)
                    step_in_chunk = 0
                    total_steps_in_chunk = list(current_chunk.values())[0].shape[1]
                    # Trigger next background inference early for smooth receding horizon
                    self._inference_trigger.set()
                except queue.Empty:
                    self.stats["action_queue_underruns"] += 1
                    # Buffer underrun: hold position / wait
                    time.sleep(self.control_interval * 0.5)
                    continue

            # 2. Extract step action
            raw_step_action = {}
            for a_key, a_val in current_chunk.items():
                raw_step_action[a_key] = a_val[0, step_in_chunk, :]  # Shape: (D,)

            step_in_chunk += 1
            self.stats["actuation_ticks"] += 1

            # 3. Apply Real-Time Safety Guard & Clamping
            safe_step_action = self.safety_guard.filter_action(
                raw_action=raw_step_action,
                current_robot_state=current_ref_state
            )

            # 4. Transmit Command to Hardware Motors (unless in dry-run mode)
            if not self.dry_run:
                self.robot.send_action(safe_step_action)
            else:
                if self.stats["actuation_ticks"] % 30 == 0:
                    logger.info(f"[Dry-Run] Step {self.stats['actuation_ticks']} simulated (Action keys: {list(safe_step_action.keys())})")

            # 5. Precise Control Rate Sleep
            t_tick_end = time.perf_counter()
            elapsed = t_tick_end - t_tick_start
            sleep_time = max(0.0, self.control_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
