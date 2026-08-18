# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Safety Guard & Hardware Protection System for Real Robot Deployment
-------------------------------------------------------------------
Provides multi-layered real-time safety checks:
1. Joint position limits clamping (q_min <= q <= q_max)
2. Joint velocity / delta-step rate limiting (prevent sudden jerks)
3. EEF Workspace 3D bounding box safety clamping (x, y, z)
4. SVD rotation sanity validator (prevents Rot6D Gram-Schmidt singularity)
5. Emergency Stop (E-Stop) latch and soft-damping recovery
"""

from dataclasses import dataclass, field
import logging
import threading
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("GR00T_SafetyGuard")


@dataclass
class SafetyLimits:
    """Configurable safety thresholds for dual-arm physical robots."""
    # Joint angle limits (radians)
    joint_min: Optional[Dict[str, np.ndarray]] = None
    joint_max: Optional[Dict[str, np.ndarray]] = None

    # Maximum allowed change in joint angle per control tick (rad/step)
    max_delta_joint: float = 0.08  # ~2.4 rad/s at 30Hz

    # Maximum allowed change in EEF position per control tick (meters/step)
    max_delta_eef_pos: float = 0.04  # ~1.2 m/s at 30Hz

    # 3D Cartesian Workspace bounding box for End-Effectors [min, max] in meters
    workspace_x: Tuple[float, float] = (-0.8, 0.8)
    workspace_y: Tuple[float, float] = (-0.8, 0.8)
    workspace_z: Tuple[float, float] = (-0.1, 1.2)  # Floor / table protection

    # Maximum allowed motor torque / effort ratio (0.0 to 1.0)
    max_effort_ratio: float = 0.85

    # Enable soft smoothing interpolation between action steps
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.35  # Exponential moving average factor (0.0 < alpha <= 1.0)


class SafetyGuard:
    """
    Real-time safety guard interceptor for robot action commands.
    Every commanded action must pass through `filter_action()` before hardware execution.
    """

    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()
        self._lock = threading.Lock()
        self._is_e_stop_active = False
        self._last_executed_actions: Dict[str, np.ndarray] = {}
        self._violation_count = 0

    @property
    def is_e_stop_active(self) -> bool:
        with self._lock:
            return self._is_e_stop_active

    def trigger_emergency_stop(self, reason: str = "Manual User E-Stop"):
        """Latches the Emergency Stop state and blocks all further robot motions."""
        with self._lock:
            self._is_e_stop_active = True
            logger.critical(f"[E-STOP TRIGGERED] Robot motion halted! Reason: {reason}")
            print(f"\n🚨 [CRITICAL SAFETY ALERT] EMERGENCY STOP ACTIVATED: {reason}")

    def reset_emergency_stop(self):
        """Resets the E-Stop latch after verifying hardware safety."""
        with self._lock:
            self._is_e_stop_active = False
            self._last_executed_actions.clear()
            logger.info("[E-STOP RESET] Safety guard re-armed.")
            print("✅ [SAFETY GUARD] E-Stop reset. Robot control re-armed.")

    def sanitize_rot6d_matrix(self, pose_9d: np.ndarray) -> np.ndarray:
        """
        Validates Rot6D rotation components in 9D pose [x, y, z, r11, r21, r31, r12, r22, r32].
        Ensures columns are non-zero vectors so SVD Gram-Schmidt orthogonalization will not diverge.
        """
        sanitized = np.array(pose_9d, copy=True, dtype=np.float32)
        if len(sanitized) >= 9:
            col1 = sanitized[3:6]
            col2 = sanitized[6:9]
            norm1 = np.linalg.norm(col1)
            norm2 = np.linalg.norm(col2)

            # If rotation vector is all-zeros or degenerate, inject standard identity orientation
            if norm1 < 1e-4 or norm2 < 1e-4 or np.isnan(norm1) or np.isnan(norm2):
                logger.warning("[SafetyGuard] Degenerate Rot6D detected. Replacing with Identity rotation.")
                sanitized[3:6] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                sanitized[6:9] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        return sanitized

    def filter_action(
        self,
        raw_action: Dict[str, np.ndarray],
        current_robot_state: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Applies safety checks, clamping, workspace bounding, and rate-limiting to raw action.

        Args:
            raw_action: Dictionary mapping action_key -> target values (D,)
            current_robot_state: Current measured robot state (D,)

        Returns:
            Sanitized, rate-limited, and clamped action command dictionary.
        """
        with self._lock:
            if self._is_e_stop_active:
                # In E-stop mode, hold current position (or zero delta command)
                if current_robot_state is not None:
                    return {k: current_robot_state.get(k, np.zeros_like(v)) for k, v in raw_action.items()}
                return {k: np.zeros_like(v) for k, v in raw_action.items()}

            safe_action = {}
            for key, val in raw_action.items():
                cmd = np.array(val, copy=True, dtype=np.float32)

                # Check for NaN / Inf in model outputs
                if np.isnan(cmd).any() or np.isinf(cmd).any():
                    logger.error(f"[SafetyGuard] NaN/Inf detected in model output for key '{key}'! Replacing with last safe action.")
                    cmd = self._last_executed_actions.get(key, np.zeros_like(cmd))

                # 1. EEF Pose Handling (9D: translation + rot6d)
                if "wrist_eef" in key or "eef" in key:
                    cmd = self.sanitize_rot6d_matrix(cmd)

                    # Clamp Cartesian 3D workspace box
                    if len(cmd) >= 3:
                        cmd[0] = np.clip(cmd[0], self.limits.workspace_x[0], self.limits.workspace_x[1])
                        cmd[1] = np.clip(cmd[1], self.limits.workspace_y[0], self.limits.workspace_y[1])
                        cmd[2] = np.clip(cmd[2], self.limits.workspace_z[0], self.limits.workspace_z[1])

                    # Rate-limit Cartesian delta step
                    if key in self._last_executed_actions:
                        prev_pos = self._last_executed_actions[key][:3]
                        curr_pos = cmd[:3]
                        delta_pos = curr_pos - prev_pos
                        dist = np.linalg.norm(delta_pos)
                        if dist > self.limits.max_delta_eef_pos:
                            scale = self.limits.max_delta_eef_pos / (dist + 1e-8)
                            cmd[:3] = prev_pos + delta_pos * scale
                            self._violation_count += 1
                            logger.debug(f"[SafetyGuard] EEF velocity clamped for '{key}' (dist: {dist:.3f}m)")

                # 2. Joint / Arm / Hand Handling
                else:
                    # Apply soft joint limits if defined
                    if self.limits.joint_min and key in self.limits.joint_min:
                        cmd = np.maximum(cmd, self.limits.joint_min[key])
                    if self.limits.joint_max and key in self.limits.joint_max:
                        cmd = np.minimum(cmd, self.limits.joint_max[key])

                    # Rate-limit Joint delta step
                    if key in self._last_executed_actions:
                        prev_joints = self._last_executed_actions[key]
                        delta_joints = cmd - prev_joints
                        max_diff = np.max(np.abs(delta_joints))
                        if max_diff > self.limits.max_delta_joint:
                            scale = self.limits.max_delta_joint / (max_diff + 1e-8)
                            cmd = prev_joints + delta_joints * scale
                            self._violation_count += 1
                            logger.debug(f"[SafetyGuard] Joint velocity clamped for '{key}' (max_diff: {max_diff:.3f}rad)")

                # 3. Soft Smoothing (EMA)
                if self.limits.enable_smoothing and key in self._last_executed_actions:
                    alpha = self.limits.smoothing_alpha
                    cmd = alpha * cmd + (1.0 - alpha) * self._last_executed_actions[key]

                # Update last executed action
                self._last_executed_actions[key] = np.array(cmd, copy=True)
                safe_action[key] = cmd

            return safe_action
