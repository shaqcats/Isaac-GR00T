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
import datetime
import json
from pathlib import Path
import time
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch

from gr00t.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag


def str2bool(v):
    """Parses boolean command line arguments (e.g. true/false/1/0/yes/no)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")


def parse_save_action_json(v):
    """Parses --save-action-json argument supporting True/False/true/false or custom file path."""
    if v is None:
        return True
    if isinstance(v, bool):
        return v
    v_str = str(v).strip()
    if v_str.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v_str.lower() in ("no", "false", "f", "n", "0"):
        return False
    # If a file path string is passed directly
    return v_str


# ==============================================================================
# Helper Function: Save Input Video Frames to Disk
# ==============================================================================
def save_video_frames_to_disk(
    video_dict: Dict[str, np.ndarray],
    output_dir: str,
    step_idx: int = 0,
    timestamp_str: str = None,
) -> Dict[str, list[str]]:
    """
    Saves the camera images from video observations into image files (PNG).

    Args:
        video_dict: Dictionary mapping video key -> np.ndarray of shape (B, T, H, W, C) or (H, W, C)
        output_dir: Directory path where images should be saved
        step_idx: Control loop iteration or step index
        timestamp_str: Optional timestamp string for file naming

    Returns:
        Dictionary mapping video key to list of saved image file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if timestamp_str is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = {}
    for key, val in video_dict.items():
        # Sanitize key name for safe filename (e.g., ego_view_res320x240_freq20 -> ego_view_res320x240_freq20)
        safe_key = key.replace("/", "_").replace(":", "_")
        saved_files[key] = []

        if isinstance(val, np.ndarray):
            # Extract list of frames based on tensor dimensionality
            if val.ndim == 5:
                # Shape (B, T, H, W, C) -> take first batch
                frames = val[0]
            elif val.ndim == 4:
                # Shape (T, H, W, C)
                frames = val
            elif val.ndim == 3:
                # Shape (H, W, C)
                frames = [val]
            else:
                continue

            for t_idx, frame in enumerate(frames):
                # Ensure uint8 type for PIL
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                img = Image.fromarray(frame)
                img_filename = f"{safe_key}_step{step_idx}_t{t_idx}_{timestamp_str}.png"
                img_path = out_path / img_filename
                img.save(img_path)
                saved_files[key].append(str(img_path.resolve()))
                print(f"[Frame Saved] Camera '{key}' (t={t_idx}) -> {img_path.resolve()}")

    return saved_files


# ==============================================================================
# Helper Function: Save Action Chunk to JSON
# ==============================================================================
def save_action_chunk_to_json(
    action_dict: Dict[str, np.ndarray],
    output_path: str,
    metadata: Dict[str, Any],
) -> Path:
    """
    Saves the predicted action chunk dictionary and metadata into a formatted JSON file.

    Args:
        action_dict: Dictionary mapping action key -> np.ndarray of shape (B, T, D)
        output_path: Target JSON file path or output directory
        metadata: Additional metadata (model path, instruction, timings, etc.)

    Returns:
        Path object of the saved JSON file.
    """
    path = Path(output_path)
    if path.is_dir() or output_path.endswith("/") or output_path.endswith("\\"):
        path.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / f"action_chunk_{timestamp_str}.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Convert NumPy arrays to nested Python lists for JSON serialization
    formatted_actions = {}
    for key, val in action_dict.items():
        if isinstance(val, np.ndarray):
            # For batch size B=1, save as (T, D) for clean readability
            if val.ndim == 3 and val.shape[0] == 1:
                formatted_actions[key] = val[0].tolist()
            else:
                formatted_actions[key] = val.tolist()
        else:
            formatted_actions[key] = val

    data_to_save = {
        "metadata": metadata,
        "action_chunk": formatted_actions,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    print(f"[JSON Saved] Predicted action chunk successfully saved to: {path.resolve()}")
    return path


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
    parser.add_argument("--save-action-json", type=parse_save_action_json, nargs="?", const=True, default=False, help="Whether to save inferred action chunk as JSON file (True/False, or custom file path)")
    parser.add_argument("--save-action-json-path", type=str, default="output_actions/predicted_actions.json", help="Target path or directory to save action JSON when --save-action-json is True")
    parser.add_argument("--save-video-frames", type=str2bool, nargs="?", const=True, default=False, help="Whether to save input camera video frames to disk (True/False)")
    parser.add_argument("--save-frames-dir", type=str, default="output_frames", help="Directory to save camera images when --save-video-frames is True")
    args = parser.parse_args()

    # 1. Resolve JSON & Video frame save settings
    if isinstance(args.save_action_json, str):
        save_json_enabled = True
        json_save_path = args.save_action_json
    elif isinstance(args.save_action_json, bool):
        save_json_enabled = args.save_action_json
        json_save_path = args.save_action_json_path if save_json_enabled else None
    else:
        save_json_enabled = False
        json_save_path = None

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    print(f"=== Initializing GR00T Policy ===")
    print(f" Model Path         : {args.model_path}")
    print(f" Embodiment         : {tag.name} (value: {tag.value})")
    print(f" Device             : {args.device}")
    print(f" Instruction        : '{args.task_instruction}'")
    print(f" Save Action JSON   : {save_json_enabled}" + (f" (Path: {json_save_path})" if save_json_enabled else ""))
    print(f" Save Video Frames  : {args.save_video_frames} (Dir: {args.save_frames_dir})")

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

            # Optional: Save Input Video Frames to disk if enabled
            saved_frames_info = {}
            if args.save_video_frames:
                saved_frames_info = save_video_frames_to_disk(
                    video_dict=video_input,
                    output_dir=args.save_frames_dir,
                    step_idx=step_loop,
                )

            # ------------------------------------------------------------------
            # Step B: Model Inference (Predict Action Chunk)
            # ------------------------------------------------------------------
            print(f"[Observation Check] Video shape: {video_input[req_video_keys[0]].shape}, State shape: {state_input[req_state_keys[0]].shape}")
            action, info = policy.get_action(observation)
            t_infer = time.time() - t_start

            # ------------------------------------------------------------------
            # Step C: Action Execution & Optional JSON Export
            # ------------------------------------------------------------------
            pred_steps = list(action.values())[0].shape[1]
            steps_to_exec = min(execution_horizon, pred_steps)
            print(f"[Inference Success] Inferred action chunk shape: (1, {pred_steps}, D) in {t_infer*1000:.1f} ms. Executing {steps_to_exec} steps...")

            # Save Action Chunk to JSON if requested
            if save_json_enabled and json_save_path:
                action_shapes = {k: list(v.shape) for k, v in action.items()}
                metadata = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model_path": args.model_path,
                    "embodiment_tag": tag.name,
                    "embodiment_value": tag.value,
                    "task_instruction": task_prompt,
                    "inference_time_ms": round(t_infer * 1000, 2),
                    "action_horizon": pred_steps,
                    "execution_horizon": steps_to_exec,
                    "action_shapes": action_shapes,
                }
                if saved_frames_info:
                    metadata["saved_video_frames"] = saved_frames_info
                save_action_chunk_to_json(action, json_save_path, metadata)

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
