# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main Real-Robot Deployment CLI Runner
-------------------------------------
Usage:
    source .venv/bin/activate
    source scripts/activate_spark.sh

    # 1. Safe Dry-Run Test (Simulates Robot & Cameras without physical motor movement)
    python -m real_robot.run_real_robot --embodiment-tag REAL_R1_PRO_SHARPA --dry-run True

    # 2. Live Testing with OpenCV USB Cameras & Custom Robot SDK
    python -m real_robot.run_real_robot \
        --embodiment-tag REAL_R1_PRO_SHARPA \
        --robot-backend custom \
        --camera-backend opencv \
        --camera-devices 0 2 4 \
        --dry-run False \
        --task-instruction "Pick up the target object with left arm"
"""

import argparse
import signal
import sys
import time
import torch

from gr00t.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

from real_robot.robot_interface import (
    BaseRobotHardware,
    MockDualArmRobot,
    ZeroMQDualArmRobot,
    ROS2DualArmRobot,
    CustomUserDualArmRobot,
)
from real_robot.camera_interface import (
    MultiCameraManager,
    OpenCVCameraStreamer,
    RealSenseCameraStreamer,
    MockCameraStreamer,
)
from real_robot.safety_guard import SafetyGuard, SafetyLimits
from real_robot.async_controller import RealtimeAsyncRobotController


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")


def parse_args():
    parser = argparse.ArgumentParser(description="GR00T Real Robot Deployment Runner")

    # Model & Embodiment
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.7-3B", help="Pretrained model path or HF hub ID")
    parser.add_argument("--embodiment-tag", type=str, default="REAL_R1_PRO_SHARPA", help="Embodiment Tag (REAL_R1_PRO_SHARPA, REAL_G1, XDOF, etc.)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Inference device")

    # Hardware Backends
    parser.add_argument("--robot-backend", type=str, default="mock", choices=["mock", "custom", "zmq", "ros2"], help="Robot Hardware Driver Backend")
    parser.add_argument("--robot-ip", type=str, default="192.168.123.10", help="Robot controller IP address for custom/zmq")
    parser.add_argument("--camera-backend", type=str, default="mock", choices=["mock", "opencv", "realsense"], help="Camera Driver Backend")
    parser.add_argument("--camera-devices", nargs="+", default=["0", "1", "2"], help="Device index / video path / serial for each camera key")

    # Control Parameters
    parser.add_argument("--control-freq", type=float, default=30.0, help="Real-time control loop frequency in Hz")
    parser.add_argument("--execution-horizon", type=int, default=16, help="Receding horizon execution steps per chunk (<= 40)")
    parser.add_argument("--task-instruction", type=str, default="Pick up the red mug with left arm and place it on the table", help="Task instruction")
    parser.add_argument("--dry-run", type=str2bool, nargs="?", const=True, default=False, help="Dry-Run mode: read sensors/cameras but do not command physical motors")

    # Logging & Debugging
    parser.add_argument("--save-action-json", type=str2bool, nargs="?", const=True, default=False, help="Save inferred actions to JSON")
    parser.add_argument("--save-action-json-path", type=str, default="output_actions/real_robot_actions.json", help="Path to save action JSON")
    parser.add_argument("--save-video-frames", type=str2bool, nargs="?", const=True, default=False, help="Save camera video frames")
    parser.add_argument("--save-frames-dir", type=str, default="output_frames/real_robot", help="Directory to save camera frames")

    return parser.parse_args()


def build_camera_manager(backend: str, required_video_keys: list[str], camera_devices: list[str]) -> MultiCameraManager:
    """Builds camera streamers for each required policy video key."""
    camera_map = {}
    for i, v_key in enumerate(required_video_keys):
        dev = camera_devices[i] if i < len(camera_devices) else str(i)
        w, h = (320, 240) if "320x240" in v_key else (224, 224)

        if backend == "opencv":
            camera_map[v_key] = OpenCVCameraStreamer(camera_id=dev, width=w, height=h, fps=30)
        elif backend == "realsense":
            camera_map[v_key] = RealSenseCameraStreamer(camera_id=dev, width=w, height=h, fps=30)
        else:
            camera_map[v_key] = MockCameraStreamer(camera_id=f"mock_{i}", width=w, height=h, fps=30)

    return MultiCameraManager(camera_map=camera_map)


def build_robot_hardware(backend: str, robot_ip: str) -> BaseRobotHardware:
    """Instantiates selected robot hardware interface."""
    if backend == "custom":
        return CustomUserDualArmRobot(robot_ip=robot_ip)
    elif backend == "zmq":
        return ZeroMQDualArmRobot(endpoint=f"tcp://{robot_ip}:5555")
    elif backend == "ros2":
        return ROS2DualArmRobot()
    else:
        return MockDualArmRobot()


def main():
    args = parse_args()

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    print(f"=== Initializing GR00T Real Robot Policy ===")
    print(f" Model Path         : {args.model_path}")
    print(f" Embodiment         : {tag.name} (value: {tag.value})")
    print(f" Device             : {args.device}")
    print(f" Robot Backend      : {args.robot_backend}")
    print(f" Camera Backend     : {args.camera_backend}")
    print(f" Dry-Run Mode       : {args.dry_run}")
    print(f" Instruction        : '{args.task_instruction}'")

    # 1. Load GR00T Policy
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=tag,
        device=args.device,
        strict=True,
    )

    req_video_keys = policy.get_modality_config()["video"].modality_keys

    # 2. Build Camera Manager & Robot Hardware
    camera_manager = build_camera_manager(
        backend=args.camera_backend,
        required_video_keys=req_video_keys,
        camera_devices=args.camera_devices,
    )

    robot = build_robot_hardware(
        backend=args.robot_backend,
        robot_ip=args.robot_ip,
    )

    # 3. Build Safety Guard
    safety_guard = SafetyGuard(limits=SafetyLimits())

    # 4. Instantiate Asynchronous Controller
    controller = RealtimeAsyncRobotController(
        policy=policy,
        robot=robot,
        camera_manager=camera_manager,
        safety_guard=safety_guard,
        control_freq=args.control_freq,
        execution_horizon=args.execution_horizon,
        task_instruction=args.task_instruction,
        dry_run=args.dry_run,
        save_action_json=args.save_action_json,
        save_action_json_path=args.save_action_json_path,
        save_video_frames=args.save_video_frames,
        save_frames_dir=args.save_frames_dir,
    )

    # Handle Ctrl+C cleanly
    def signal_handler(sig, frame):
        print("\n[Signal] Ctrl+C received. Stopping controller safely...")
        controller.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start Real-Time Loop
    controller.start()

    try:
        while controller.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        controller.stop()


if __name__ == "__main__":
    main()
