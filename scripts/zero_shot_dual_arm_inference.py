# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Isaac GR00T N1.7 Dual-Arm Robot Zero-Shot Inference Script (DGX Spark Ready)
---------------------------------------------------------------------------
This script demonstrates how to run zero-shot VLA inference for a custom dual-arm robot
using the pretrained base model `nvidia/GR00T-N1.7-3B` and pretrain embodiment tags.

DGX Spark / Platform Environment Activation (Do NOT use `uv run` on DGX Spark):
    source .venv/bin/activate
    source scripts/activate_spark.sh

Usage:
    python scripts/zero_shot_dual_arm_inference.py --model-path nvidia/GR00T-N1.7-3B --embodiment-tag REAL_G1
"""

import argparse
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch

from gr00t.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag


# ==============================================================================
# [USER CONFIG 1] Embodiment & Model Configuration (기본 모델 및 태그 설정)
# ==============================================================================
# Base model for zero-shot inference: "nvidia/GR00T-N1.7-3B" or local checkpoint path
DEFAULT_MODEL_PATH = "nvidia/GR00T-N1.7-3B"

# Available dual-arm / humanoid pretrain embodiment tags:
#  - EmbodimentTag.REAL_G1 ("real_g1_relative_eef_relative_joints") -> Unitree G1 dual-arm humanoid
#  - EmbodimentTag.REAL_R1_PRO_SHARPA ("real_r1_pro_sharpa_relative_eef") -> R1 Pro Sharpa dual-arm
#  - EmbodimentTag.XDOF ("xdof_relative_eef_relative_joint") -> Generic X-DOF relative EEF+joint
DEFAULT_EMBODIMENT_TAG = EmbodimentTag.REAL_G1


# ==============================================================================
# [USER CONFIG 2] Real Robot HW Dummy Interface (사용자 로봇 hardware SDK 바인딩)
# ==============================================================================
class CustomDualArmRobot:
    """
    Dummy/Wrapper class for user's dual-arm robot hardware SDK.
    Replace these methods with your real robot control SDK (e.g. Unitree SDK, ROS2, ZMQ, etc.).
    """

    def __init__(self):
        print("[Robot] Initializing Dual-Arm Hardware Interface...")
        # TODO: Initialize camera drivers (e.g., OpenCV, RealSense SDK)
        # TODO: Initialize robot arm joints & grippers SDK

    def capture_cameras(self) -> Dict[str, np.ndarray]:
        """
        [USER CONFIG 2-A] Capture live camera frames.
        Must return RGB uint8 numpy arrays with shape (H, W, 3) and values in [0, 255].
        """
        # Example dummy camera frame 224x224 RGB
        dummy_ego_cam = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Match camera keys based on selected EmbodimentTag:
        #  - For REAL_G1: "ego_view"
        #  - For REAL_R1_PRO_SHARPA / DROID: "exterior_image_1_left", "wrist_image_left"
        return {
            "ego_view": dummy_ego_cam,
        }

    def get_joint_states(self) -> Dict[str, np.ndarray]:
        """
        [USER CONFIG 2-B] Read current joint positions / states.
        Must return float32 numpy arrays with shape (D,).
        """
        # Example joint positions for Unitree G1 dual-arm structure:
        # Left arm (7 joints), Right arm (7 joints), Left hand (7 joints), Right hand (7 joints), Waist (3 joints)
        return {
            "left_leg": np.zeros(6, dtype=np.float32),
            "right_leg": np.zeros(6, dtype=np.float32),
            "waist": np.zeros(3, dtype=np.float32),
            "left_arm": np.zeros(7, dtype=np.float32),
            "right_arm": np.zeros(7, dtype=np.float32),
            "left_hand": np.zeros(7, dtype=np.float32),
            "right_hand": np.zeros(7, dtype=np.float32),
        }

    def execute_action(self, action_step: Dict[str, np.ndarray]):
        """
        [USER CONFIG 2-C] Send target joint commands to dual-arm hardware.
        """
        # Extract actions for left & right arms
        # left_arm_cmd = action_step["left_arm"]
        # right_arm_cmd = action_step["right_arm"]
        # print(f"[Robot Execute] Left arm target: {left_arm_cmd[:3]}... Right arm target: {right_arm_cmd[:3]}...")
        pass


# ==============================================================================
# Main Zero-Shot Inference Control Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="GR00T N1.7 Zero-Shot Dual-Arm Policy Inference")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to base model or HuggingFace ID")
    parser.add_argument("--embodiment-tag", type=str, default="REAL_G1", help="Pretrain embodiment tag (e.g. REAL_G1, XDOF)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--execution-horizon", type=int, default=16, help="Number of action steps to execute per inference cycle (<= 40)")
    parser.add_argument("--task-instruction", type=str, default="Pick up the red mug with left arm and place it on the right table", help="Natural language instruction for the robot")
    args = parser.parse_args()

    # 1. Resolve Embodiment Tag
    tag = EmbodimentTag.resolve(args.embodiment_tag)
    print(f"=== Initializing GR00T Policy ===")
    print(f" Model Path    : {args.model_path}")
    print(f" Embodiment    : {tag.name} (value: {tag.value})")
    print(f" Device        : {args.device}")
    print(f" Instruction   : '{args.task_instruction}'")

    # 2. Instantiate Policy
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=tag,
        device=args.device,
        strict=True,
    )

    # 3. Inspect Expected Modality Configs
    modality_configs = policy.get_modality_config()
    print("\n=== Expected Observation Modality Config ===")
    print("Video Keys:", modality_configs["video"].modality_keys)
    print("State Keys:", modality_configs["state"].modality_keys)
    print("Action Keys:", modality_configs["action"].modality_keys)

    # 4. Instantiate Robot Interface
    robot = CustomDualArmRobot()

    # ==========================================================================
    # [USER CONFIG 3] Language & Execution Parameters
    # ==========================================================================
    task_prompt = args.task_instruction
    execution_horizon = args.execution_horizon  # e.g. execute 16 steps out of predicted 40 steps chunk

    print("\n=== Starting Closed-Loop Zero-Shot Control Loop ===")
    try:
        while True:
            t_start = time.time()

            # ------------------------------------------------------------------
            # Step A: Capture Robot Observations
            # ------------------------------------------------------------------
            camera_dict = robot.capture_cameras()
            state_dict = robot.get_joint_states()

            # Format Video Inputs: Shape (B=1, T=1, H, W, C=3), uint8
            # Note: T depends on delta_indices in modality_config (usually 1 or 2 timesteps)
            video_input = {}
            for v_key in modality_configs["video"].modality_keys:
                if v_key in camera_dict:
                    raw_img = camera_dict[v_key]  # (H, W, 3)
                    video_input[v_key] = np.expand_dims(np.expand_dims(raw_img, axis=0), axis=0) # (1, 1, H, W, 3)
                else:
                    # Fallback dummy image if key missing
                    video_input[v_key] = np.zeros((1, 1, 224, 224, 3), dtype=np.uint8)

            # Format State Inputs: Shape (B=1, T=1, D), float32
            state_input = {}
            for s_key in modality_configs["state"].modality_keys:
                if s_key in state_dict:
                    raw_state = state_dict[s_key]
                    state_input[s_key] = np.expand_dims(np.expand_dims(raw_state, axis=0), axis=0).astype(np.float32)
                else:
                    # Dummy zero state fallback for missing keys
                    state_input[s_key] = np.zeros((1, 1, 1), dtype=np.float32)

            # Format Language Inputs: List of list of strings, shape (B=1, T=1)
            lang_key = modality_configs["language"].modality_keys[0]
            language_input = {
                lang_key: [[task_prompt]]
            }

            observation = {
                "video": video_input,
                "state": state_input,
                "language": language_input,
            }

            # ------------------------------------------------------------------
            # Step B: Model Inference (Predict Action Chunk)
            # ------------------------------------------------------------------
            action, info = policy.get_action(observation)
            t_infer = time.time() - t_start

            # ------------------------------------------------------------------
            # Step C: Action Execution (Receding Horizon Execution)
            # ------------------------------------------------------------------
            # Each action value has shape (B=1, T_pred=40, D)
            pred_steps = list(action.values())[0].shape[1]
            steps_to_exec = min(execution_horizon, pred_steps)
            print(f"[Inference] Inferred action chunk shape: (1, {pred_steps}, D) in {t_infer*1000:.1f} ms. Executing {steps_to_exec} steps...")

            for step_idx in range(steps_to_exec):
                current_action_step = {}
                for a_key, a_val in action.items():
                    current_action_step[a_key] = a_val[0, step_idx, :]  # Shape: (D,)

                robot.execute_action(current_action_step)
                time.sleep(1.0 / 30.0)  # 30 FPS robot execution rate

    except KeyboardInterrupt:
        print("\n[Control Loop] Stopped by user.")

if __name__ == "__main__":
    main()
