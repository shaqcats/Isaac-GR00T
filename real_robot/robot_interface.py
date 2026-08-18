# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Robot Hardware Abstraction Layer (HAL)
--------------------------------------
Defines base contract for physical dual-arm / humanoid robot hardware interfaces
with concrete implementations for Mock, ROS2, ZeroMQ, and Custom User SDKs.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger("GR00T_RobotHAL")


class BaseRobotHardware(ABC):
    """
    Abstract Base Class for Dual-Arm & Humanoid Robot Hardware Drivers.
    Any real robot (Unitree G1, Real R1 Pro Sharpa, Franka, AgileX, etc.)
    inherits from this class and implements `connect`, `get_state`, and `send_action`.
    """

    def __init__(self, name: str = "RobotHAL"):
        self.name = name
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """Establishes connection to physical robot motors / CAN / network bus."""
        pass

    @abstractmethod
    def disconnect(self):
        """Safely stops motion and disconnects from hardware."""
        pass

    @abstractmethod
    def get_state(self, state_keys: List[str], state_dims: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        """
        Reads latest joint positions, velocities, and EEF poses from hardware sensors.

        Args:
            state_keys: List of expected state keys by policy (e.g. 'left_wrist_eef', 'left_hand_joints')
            state_dims: Optional dictionary mapping key to expected dimension D.

        Returns:
            Dictionary mapping state key -> float32 numpy array with shape (D,)
        """
        pass

    @abstractmethod
    def send_action(self, action_step: Dict[str, np.ndarray]) -> bool:
        """
        Transmits target action command for one control tick to the physical robot.

        Args:
            action_step: Dictionary mapping action key -> target command (D,)

        Returns:
            True if command successfully accepted by hardware controller.
        """
        pass


class MockDualArmRobot(BaseRobotHardware):
    """
    High-fidelity simulated mock robot for safe offline testing,
    latency profiling, and pipeline verification without real motors.
    """

    def __init__(self, name: str = "MockDualArmRobot"):
        super().__init__(name)
        self._current_state: Dict[str, np.ndarray] = {}

    def connect(self) -> bool:
        self._connected = True
        logger.info(f"[{self.name}] Mock robot connected and ready.")
        print(f"🤖 [{self.name}] Virtual robot interface connected.")
        return True

    def disconnect(self):
        self._connected = False
        logger.info(f"[{self.name}] Mock robot disconnected.")

    def get_state(self, state_keys: List[str], state_dims: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        state_dims = state_dims or {}
        for key in state_keys:
            if key not in self._current_state:
                dim = state_dims.get(key, 9 if "wrist_eef" in key else 22 if "hand_joints" in key else 7)
                st = np.zeros(dim, dtype=np.float32)
                if "wrist_eef" in key:
                    # Identity Rot6D pose [x, y, z, r11=1, r21=0, r31=0, r12=0, r22=1, r32=0]
                    st[0] = 0.35  # Default EEF position forward
                    st[1] = 0.20 if "left" in key else -0.20
                    st[2] = 0.45
                    st[3] = 1.0  # r11 = 1.0
                    st[7] = 1.0  # r22 = 1.0
                self._current_state[key] = st

        return {k: np.array(v, copy=True, dtype=np.float32) for k, v in self._current_state.items() if k in state_keys}

    def send_action(self, action_step: Dict[str, np.ndarray]) -> bool:
        # Simulate smooth physical motion by updating current state
        for key, cmd in action_step.items():
            if key in self._current_state:
                # First-order response simulation
                self._current_state[key] = 0.8 * self._current_state[key] + 0.2 * cmd
            else:
                self._current_state[key] = np.array(cmd, copy=True, dtype=np.float32)
        return True


class ZeroMQDualArmRobot(BaseRobotHardware):
    """
    ZeroMQ IPC/TCP Socket Client for interfacing with external C++ / Python robot control daemons.
    """

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", name: str = "ZeroMQ_Robot"):
        super().__init__(name)
        self.endpoint = endpoint
        self._socket = None
        self._context = None

    def connect(self) -> bool:
        try:
            import zmq
        except ImportError:
            raise ImportError("pyzmq is required for ZeroMQDualArmRobot. Install via: pip install pyzmq")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)
        self._socket.setsockopt(zmq.SNDTIMEO, 1000)
        self._socket.connect(self.endpoint)
        self._connected = True
        logger.info(f"[{self.name}] Connected to ZeroMQ robot daemon at {self.endpoint}")
        return True

    def disconnect(self):
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
        self._connected = False

    def get_state(self, state_keys: List[str], state_dims: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        import json
        req = {"cmd": "get_state", "keys": state_keys}
        try:
            self._socket.send_json(req)
            resp = self._socket.recv_json()
            states = {}
            for k, v in resp.get("state", {}).items():
                states[k] = np.array(v, dtype=np.float32)
            return states
        except Exception as e:
            logger.error(f"[{self.name}] ZeroMQ get_state failed: {e}")
            # Fallback safe zeros
            return {k: np.zeros(state_dims.get(k, 7), dtype=np.float32) for k in state_keys}

    def send_action(self, action_step: Dict[str, np.ndarray]) -> bool:
        serializable = {k: v.tolist() for k, v in action_step.items()}
        req = {"cmd": "send_action", "action": serializable}
        try:
            self._socket.send_json(req)
            resp = self._socket.recv_json()
            return resp.get("status") == "ok"
        except Exception as e:
            logger.error(f"[{self.name}] ZeroMQ send_action failed: {e}")
            return False


class ROS2DualArmRobot(BaseRobotHardware):
    """
    ROS 2 Hardware Bridge subscribing to sensor topics and publishing trajectory commands.
    """

    def __init__(self, name: str = "ROS2_Robot"):
        super().__init__(name)
        self._node = None
        self._executor = None

    def connect(self) -> bool:
        try:
            import rclpy
            from rclpy.node import Node
        except ImportError:
            logger.warning("[ROS2_Robot] rclpy not found. Please source ROS2 environment.")
            return False

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("gr00t_robot_bridge")
        self._connected = True
        logger.info("[ROS2_Robot] ROS2 Node initialized.")
        return True

    def disconnect(self):
        if self._node:
            self._node.destroy_node()
        self._connected = False

    def get_state(self, state_keys: List[str], state_dims: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        # TODO: Implement subscriber callback caching from /joint_states topic
        return {k: np.zeros(state_dims.get(k, 7), dtype=np.float32) for k in state_keys}

    def send_action(self, action_step: Dict[str, np.ndarray]) -> bool:
        # TODO: Implement trajectory publisher to /joint_trajectory_controller/joint_trajectory
        return True


class CustomUserDualArmRobot(BaseRobotHardware):
    """
    USER TEMPLATE: Fill in your proprietary robot vendor SDK methods here
    (e.g., Unitree SDK, Franka libfranka, Kinova Kortex, Dynamixel SDK, AgileX, etc.).
    """

    def __init__(self, robot_ip: str = "192.168.123.10", name: str = "CustomUserRobot"):
        super().__init__(name)
        self.robot_ip = robot_ip

    def connect(self) -> bool:
        print(f"[{self.name}] Connecting to physical robot at IP: {self.robot_ip}...")
        # ----------------------------------------------------------------------
        # [USER EDIT 1]: Initialize your robot SDK / CAN bus / Ethernet connection
        # Example:
        #   self.robot_sdk = MyVendorSDK.connect(self.robot_ip)
        #   self.robot_sdk.enable_motors()
        # ----------------------------------------------------------------------
        self._connected = True
        print(f"[{self.name}] ✅ Successfully connected to robot hardware.")
        return True

    def disconnect(self):
        print(f"[{self.name}] Safely powering off robot motors...")
        # ----------------------------------------------------------------------
        # [USER EDIT 2]: Safely stop and disconnect
        # Example:
        #   self.robot_sdk.disable_motors()
        # ----------------------------------------------------------------------
        self._connected = False

    def get_state(self, state_keys: List[str], state_dims: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        """
        [USER EDIT 3]: Read actual sensor feedback from physical robot.
        Must return Dict[str, np.ndarray] where values are 1D float32 numpy arrays (D,).
        """
        states = {}
        state_dims = state_dims or {}

        # Example hardware feedback mapping:
        # left_arm_q = self.robot_sdk.get_left_arm_joint_positions()   # shape: (7,)
        # right_arm_q = self.robot_sdk.get_right_arm_joint_positions() # shape: (7,)
        # left_hand_q = self.robot_sdk.get_left_hand_joint_positions() # shape: (22,)

        for key in state_keys:
            dim = state_dims.get(key, 9 if "wrist_eef" in key else 7)
            if "wrist_eef" in key:
                # 9D Pose: [x, y, z, rot6d_r11, rot6d_r21, rot6d_r31, rot6d_r12, rot6d_r22, rot6d_r32]
                st = np.zeros(dim, dtype=np.float32)
                st[3] = 1.0  # r11
                st[7] = 1.0  # r22
                states[key] = st
            else:
                states[key] = np.zeros(dim, dtype=np.float32)

        return states

    def send_action(self, action_step: Dict[str, np.ndarray]) -> bool:
        """
        [USER EDIT 4]: Transmit commanded action step to physical motors.
        """
        # Example command dispatch:
        # if "left_wrist_eef" in action_step:
        #     self.robot_sdk.set_left_eef_target_pose(action_step["left_wrist_eef"])
        # if "left_arm" in action_step:
        #     self.robot_sdk.set_left_arm_target_positions(action_step["left_arm"])
        # if "left_hand_joints" in action_step:
        #     self.robot_sdk.set_left_hand_joint_targets(action_step["left_hand_joints"])

        return True
