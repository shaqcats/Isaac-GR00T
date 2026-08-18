# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GR00T Real Robot Deployment Module
----------------------------------
Modular hardware abstraction, real-time camera management, safety guards,
and asynchronous policy controllers for real-world dual-arm & humanoid robots.
"""

from real_robot.robot_interface import (
    BaseRobotHardware,
    MockDualArmRobot,
    ROS2DualArmRobot,
    ZeroMQDualArmRobot,
)
from real_robot.camera_interface import (
    MultiCameraManager,
    OpenCVCameraStreamer,
    RealSenseCameraStreamer,
    MockCameraStreamer,
)
from real_robot.safety_guard import SafetyGuard, SafetyLimits
from real_robot.async_controller import RealtimeAsyncRobotController

__all__ = [
    "BaseRobotHardware",
    "MockDualArmRobot",
    "ROS2DualArmRobot",
    "ZeroMQDualArmRobot",
    "MultiCameraManager",
    "OpenCVCameraStreamer",
    "RealSenseCameraStreamer",
    "MockCameraStreamer",
    "SafetyGuard",
    "SafetyLimits",
    "RealtimeAsyncRobotController",
]
