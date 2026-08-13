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
# [USER CONFIG 2] Real Robot HW Interface (사용자 로봇 hardware SDK 바인딩)
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

    def capture_cameras(self, video_keys: list[str]) -> Dict[str, np.ndarray]:
        """
        [USER CONFIG 2-A] Capture live camera frames for requested video keys.
        Must return RGB uint8 numpy arrays with shape (H, W, 3) and values in [0, 255].
        """
        cameras = {}
        for key in video_keys:
            # Parse resolution hint if embedded in key name (e.g., "res320x240"), else default 224x224
            if "320x240" in key:
                h, w = 240, 320
            else:
                h, w = 224, 224
            
            # Generate dummy random RGB image frame (H, W, 3)
            cameras[key] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        
        return cameras

    def get_joint_states(self, state_keys: list[str], state_dims: Dict[str, int] = None) -> Dict[str, np.ndarray]:
        """
        [USER CONFIG 2-B] Read current joint positions / EEF states for requested state keys.
        Must return float32 numpy arrays with shape (D,).
        """
        states = {}
        state_dims = state_dims or {}
        for key in state_keys:
            # Determine dimension D for key
            if key in state_dims:
                dim = state_dims[key]
            elif "wrist_eef" in key:
                dim = 9
            elif "hand_joints" in key:
                dim = 22  # Real R1 Pro Sharpa 22-DoF hand joints
            elif "arm" in key or "hand" in key:
                dim = 7
            elif "leg" in key:
                dim = 6
            elif "waist" in key:
                dim = 3
            else:
                dim = 7

            if "wrist_eef" in key:
                # 9D pose: [x, y, z, rot6d_r11, rot6d_r21, rot6d_r31, rot6d_r12, rot6d_r22, rot6d_r32]
                # Set identity rot6d matrix so SVD Gram-Schmidt orthogonalization doesn't fail
                st = np.zeros(dim, dtype=np.float32)
                st[3] = 1.0  # r11 = 1.0
                st[7] = 1.0  # r22 = 1.0
                states[key] = st
            else:
                states[key] = np.zeros(dim, dtype=np.float32)

        return states

    def execute_action(self, action_step: Dict[str, np.ndarray]):
        """
        [USER CONFIG 2-C] Send target joint commands to dual-arm hardware.
        """
        # Extract actions for left & right arms / EEF
        # e.g., left_arm_cmd = action_step.get("left_arm") or action_step.get("left_wrist_eef")
        pass


# ==============================================================================
# Main Zero-Shot Inference Control Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="GR00T N1.7 Zero-Shot Dual-Arm Policy Inference")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to base model or HuggingFace ID")
    parser.add_argument("--embodiment-tag", type=str, default="REAL_R1_PRO_SHARPA", help="Pretrain embodiment tag (e.g. REAL_G1, REAL_R1_PRO_SHARPA, XDOF)")
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
    print("Video Keys                      :", modality_configs["video"].modality_keys)
    print("Video Delta Indices (Horizon T) :", modality_configs["video"].delta_indices)
    print("State Keys                      :", modality_configs["state"].modality_keys)
    print("State Delta Indices (Horizon T) :", modality_configs["state"].delta_indices)
    print("Action Keys                     :", modality_configs["action"].modality_keys)

    # Extract required keys & temporal horizons dynamically
    req_video_keys = modality_configs["video"].modality_keys
    req_state_keys = modality_configs["state"].modality_keys
    req_lang_keys = modality_configs["language"].modality_keys

    v_horizon = len(modality_configs["video"].delta_indices)  # e.g., 2 for R1 Sharpa, 1 for G1
    s_horizon = len(modality_configs["state"].delta_indices)  # e.g., 1
    l_horizon = len(modality_configs["language"].delta_indices)  # e.g., 1

    # Infer state dimensions dynamically from processor normalization statistics
    state_dims = {}
    try:
        proc_stats = policy.processor.statistics[tag.value]["state"]
        for s_key in req_state_keys:
            if s_key in proc_stats:
                state_dims[s_key] = proc_stats[s_key]["min"].shape[-1]
    except Exception as e:
        print(f"[Warning] Could not dynamically query processor state stats: {e}")

    print("\n=== Inferred State Dimensions (D) ===")
    for s_key in req_state_keys:
        print(f" - {s_key}: D={state_dims.get(s_key, 'default')}")

    # 4. Instantiate Robot Interface
    robot = CustomDualArmRobot()

    # ==========================================================================
    # [USER CONFIG 3] Language & Execution Parameters
    # ==========================================================================
    task_prompt = args.task_instruction
    execution_horizon = args.execution_horizon  # e.g. execute 16 steps out of predicted 40 steps chunk

    print(f"\n=== Starting Closed-Loop Zero-Shot Control Loop ({tag.name}) ===")
    try:
        # Run single iteration test or continuous loop
        for step_loop in range(1):  # Can be changed to `while True:` for infinite loop
            t_start = time.time()

            # ------------------------------------------------------------------
            # Step A: Capture Robot Observations dynamically based on required keys
            # ------------------------------------------------------------------
            camera_dict = robot.capture_cameras(req_video_keys)
            state_dict = robot.get_joint_states(req_state_keys, state_dims=state_dims)

            # Format Video Inputs dynamically: Shape (B=1, T=v_horizon, H, W, C=3), uint8
            video_input = {}
            for v_key in req_video_keys:
                raw_img = camera_dict[v_key]  # Shape: (H, W, 3)
                # Replicate/Stack image across the required temporal horizon T
                img_sequence = np.stack([raw_img] * v_horizon, axis=0)  # Shape: (v_horizon, H, W, 3)
                video_input[v_key] = np.expand_dims(img_sequence, axis=0) # Shape: (1, v_horizon, H, W, 3)

            # Format State Inputs dynamically: Shape (B=1, T=s_horizon, D), float32
            state_input = {}
            for s_key in req_state_keys:
                raw_state = state_dict[s_key]  # Shape: (D,)
                # Replicate/Stack state across the required temporal horizon T
                state_sequence = np.stack([raw_state] * s_horizon, axis=0)  # Shape: (s_horizon, D)
                state_input[s_key] = np.expand_dims(state_sequence, axis=0).astype(np.float32)  # Shape: (1, s_horizon, D)

            # Format Language Inputs dynamically: Shape (B=1, T=l_horizon)
            language_input = {
                req_lang_keys[0]: [[task_prompt] * l_horizon]
            }

            observation = {
                "video": video_input,
                "state": state_input,
                "language": language_input,
            }

            # ------------------------------------------------------------------
            # Step B: Model Inference (Predict Action Chunk)
            # ------------------------------------------------------------------
            print(f"[Observation Check] Video shape: {video_input[req_video_keys[0]].shape}, State shape: {state_input[req_state_keys[0]].shape}")
            action, info = policy.get_action(observation)
            t_infer = time.time() - t_start

            # ------------------------------------------------------------------
            # Step C: Action Execution (Receding Horizon Execution)
            # ------------------------------------------------------------------
            # Each action value has shape (B=1, T_pred=40, D)
            pred_steps = list(action.values())[0].shape[1]
            steps_to_exec = min(execution_horizon, pred_steps)
            print(f"[Inference Success] Inferred action chunk shape: (1, {pred_steps}, D) in {t_infer*1000:.1f} ms. Executing {steps_to_exec} steps...")

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
