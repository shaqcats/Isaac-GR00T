# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
================================================================================
🤖 Isaac GR00T N1.7 - REAL_R1_PRO_SHARPA 전용 Zero-Shot 추론 스크립트
================================================================================

본 스크립트는 R1 Pro Sharpa (양팔 + 22-DoF 다지형 로봇손) 하드웨어에
NVIDIA GR00T-N1.7 파운데이션 모델을 즉시 적용할 수 있도록 제작된 전용 코드입니다.

--------------------------------------------------------------------------------
📌 [1] 하드웨어 및 모델 규격 요약 (REAL_R1_PRO_SHARPA)
--------------------------------------------------------------------------------
1. 모델명: nvidia/GR00T-N1.7-3B
2. Embodiment Tag: REAL_R1_PRO_SHARPA (value: 'real_r1_pro_sharpa_relative_eef')
3. 입력 모달리티 (Observations):
   - 비디오 (3개 카메라, Horizon T=2 프레임 시퀀스, 해상도 320x240 RGB):
     * 'ego_view_res320x240_freq20'        : 로봇 머리/가슴 Ego 시점 카메라
     * 'left_wrist_view_res320x240_freq20' : 왼손 손목 카메라
     * 'right_wrist_view_res320x240_freq20': 오른손 손목 카메라
   - 관절 및 포즈 상태 (4개 키, Horizon T=1):
     * 'left_wrist_eef'   : 9차원 [x, y, z, rot6d (6차원)] - 왼손목 End-Effector 3D Pose
     * 'right_wrist_eef'  : 9차원 [x, y, z, rot6d (6차원)] - 오른손목 End-Effector 3D Pose
     * 'left_hand_joints' : 22차원 - 왼손 22-DoF 다지형 그리퍼 각도 (라디안 [rad])
     * 'right_hand_joints': 22차원 - 오른손 22-DoF 다지형 그리퍼 각도 (라디안 [rad])
   - 언어 (자연어 지시문):
     * 'annotation.human.coarse_action'   : 작업 명령 (예: "Pick up the red mug with left arm and place it on the right table")

4. 출력 모달리티 (Actions):
   - 미래 40단계 액션 묶음 (Action Chunk @ 30Hz, 약 1.33초 분량 궤적):
     * 'left_wrist_eef'   : Shape (1, 40, 9)  - 왼손목 목표 3D 이동량 [dx, dy, dz (m)] + Rot6D 회전
     * 'right_wrist_eef'  : Shape (1, 40, 9)  - 오른손목 목표 3D 이동량 [dx, dy, dz (m)] + Rot6D 회전
     * 'left_hand_joints' : Shape (1, 40, 22) - 왼손 22개 손가락 관절 목표 각도 [rad]
     * 'right_hand_joints': Shape (1, 40, 22) - 오른손 22개 손가락 관절 목표 각도 [rad]

--------------------------------------------------------------------------------
🚀 [2] 실행 방법 (DGX Spark or AGX Thor / Linux PC)
--------------------------------------------------------------------------------
    # Spark
    $ bash scripts/deployment/spark/install_deps.sh
    $ source .venv/bin/activate
    $ source scripts/activate_spark.sh  
    or
    # Thor
    $ bash scripts/deployment/thor/install_deps.sh
    $ source .venv/bin/activate
    $ source scripts/activate_thor.sh

    # 1. 기본 실행 (더미 센서 모드 테스트 => use_dummy = False 로 코드 수정후 상세 코드 수정)
    $ python scripts/r1_pro_sharpa_zero_shot.py

    # 2. 작업 지시문 변경 및 추론 결과 JSON 저장
    $ python scripts/r1_pro_sharpa_zero_shot.py \
          --task-instruction "Pick up the red mug with left arm and place it on the table" \
          --save-action-json True \
          --save-action-json-path output_actions/r1_sharpa_action.json

    # 3. 카메라 입력 이미지와 예측 액션을 모두 파일로 저장하여 확인
    $ python scripts/r1_pro_sharpa_zero_shot.py \
          --save-video-frames True \
          --save-frames-dir output_frames/r1_sharpa \
          --save-action-json True
"""

import argparse
import datetime
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from gr00t.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag


# ==============================================================================
# [MATH UTILS] Rot6D ↔ 쿼터니언/오일러 각도 변환 함수
# ==============================================================================
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Rot6D 6차원 벡터 [r11, r21, r31, r12, r22, r32]를 Gram-Schmidt 직교화 과정을 거쳐
    3x3 직교 회전 행렬(Rotation Matrix)로 복원합니다.
    """
    col1 = rot6d[:3]
    col2 = rot6d[3:6]
    
    # 1. First column 정규화
    c1_norm = np.linalg.norm(col1)
    if c1_norm < 1e-6:
        col1 = np.array([1.0, 0.0, 0.0])
    else:
        col1 = col1 / c1_norm
        
    # 2. Second column 직교화 및 정규화
    col2 = col2 - np.dot(col1, col2) * col1
    c2_norm = np.linalg.norm(col2)
    if c2_norm < 1e-6:
        col2 = np.array([0.0, 1.0, 0.0])
    else:
        col2 = col2 / c2_norm
        
    # 3. Third column은 외적(Cross product)으로 계산
    col3 = np.cross(col1, col2)
    
    return np.column_stack([col1, col2, col3])


def rot6d_to_quaternion(rot6d: np.ndarray) -> np.ndarray:
    """
    Rot6D 6차원 벡터를 쿼터니언 [qx, qy, qz, qw] (SciPy / ROS 표준)으로 변환합니다.
    """
    matrix = rot6d_to_matrix(rot6d)
    return Rotation.from_matrix(matrix).as_quat()


def rot6d_to_euler(rot6d: np.ndarray, seq: str = "xyz", degrees: bool = True) -> np.ndarray:
    """
    Rot6D 6차원 벡터를 오일러 각도 [Roll, Pitch, Yaw] (기본 단위: 도[deg])로 변환합니다.
    """
    matrix = rot6d_to_matrix(rot6d)
    return Rotation.from_matrix(matrix).as_euler(seq, degrees=degrees)


# ==============================================================================
# 🤖 [USER HARDWARE INTERFACE] REAL_R1_PRO_SHARPA 로봇 하드웨어 연동 클래스
# ==============================================================================
class R1ProSharpaRobot:
    """
    REAL_R1_PRO_SHARPA 로봇(양팔 + 22-DoF 손) 전용 하드웨어 연동 인터페이스.
    
    실제 로봇에 연결할 때는 __init__, capture_cameras, get_robot_states, execute_action 내부의
    TODO 부분에 실제 R1 Pro 로봇 SDK 및 카메라 드라이버를 연결하세요.
    """

    # R1 Pro Sharpa가 요구하는 정확한 카메라 및 관절 키 목록
    VIDEO_KEYS = [
        "ego_view_res320x240_freq20",        # 머리/가슴 Ego 시점 (320x240)
        "left_wrist_view_res320x240_freq20", # 왼손 손목 카메라 (320x240)
        "right_wrist_view_res320x240_freq20" # 오른손 손목 카메라 (320x240)
    ]
    
    STATE_KEYS = [
        "left_wrist_eef",   # 9D [x, y, z, rot6d (6D)]
        "right_wrist_eef",  # 9D [x, y, z, rot6d (6D)]
        "left_hand_joints", # 22D [22개 손가락 관절 각도 (rad)]
        "right_hand_joints" # 22D [22개 손가락 관절 각도 (rad)]
    ]

    def __init__(self, use_dummy: bool = True):
        self.use_dummy = use_dummy
        print(f"[R1ProSharpaRobot] Initializing Hardware Interface (Dummy Mode: {self.use_dummy})...")
        
        if not self.use_dummy:
            # TODO: 실제 R1 Pro Sharpa 로봇 SDK 및 카메라 초기화
            # 예: self.camera_ego = RealSenseCamera(serial="...")
            # 예: self.camera_left_wrist = RealSenseCamera(serial="...")
            # 예: self.camera_right_wrist = RealSenseCamera(serial="...")
            # 예: self.robot_client = R1ProRobotSDK(ip="192.168.1.100")
            pass

    def capture_cameras(self) -> Dict[str, np.ndarray]:
        """
        [입력 1] R1 Pro Sharpa의 3개 카메라로부터 최신 프레임 획득
        
        반환 규격:
            Dict[str, np.ndarray]:
              - 'ego_view_res320x240_freq20'        : (240, 320, 3) RGB uint8 이미지 [0 ~ 255]
              - 'left_wrist_view_res320x240_freq20' : (240, 320, 3) RGB uint8 이미지 [0 ~ 255]
              - 'right_wrist_view_res320x240_freq20': (240, 320, 3) RGB uint8 이미지 [0 ~ 255]
        """
        cameras = {}
        for key in self.VIDEO_KEYS:
            if self.use_dummy:
                # 더미 랜덤 240x320 RGB 이미지 생성
                cameras[key] = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            else:
                # TODO: 실제 카메라 드라이버로부터 320x240 RGB 이미지 획득
                # if key == "ego_view_res320x240_freq20":
                #     cameras[key] = self.camera_ego.get_rgb_frame() # shape (240, 320, 3)
                # elif key == "left_wrist_view_res320x240_freq20":
                #     cameras[key] = self.camera_left_wrist.get_rgb_frame()
                # elif key == "right_wrist_view_res320x240_freq20":
                #     cameras[key] = self.camera_right_wrist.get_rgb_frame()
                raise NotImplementedError("실제 카메라 드라이버 연동 코드를 작성하세요.")

        return cameras

    def get_robot_states(self) -> Dict[str, np.ndarray]:
        """
        [입력 2] R1 Pro Sharpa 로봇의 현재 End-Effector 포즈 및 손가락 관절 각도 획득
        
        반환 규격:
            Dict[str, np.ndarray]:
              - 'left_wrist_eef'   : (9,) float32 [X, Y, Z (m), Rot6D_r11, r21, r31, r12, r22, r32]
              - 'right_wrist_eef'  : (9,) float32 [X, Y, Z (m), Rot6D_r11, r21, r31, r12, r22, r32]
              - 'left_hand_joints' : (22,) float32 [22개 관절 각도 (rad)]
              - 'right_hand_joints': (22,) float32 [22개 관절 각도 (rad)]
        """
        states = {}
        
        if self.use_dummy:
            # 1. 왼손목 EEF 9D Pose (단위 회전 행렬 r11=1.0, r22=1.0 적용)
            left_eef = np.zeros(9, dtype=np.float32)
            left_eef[3] = 1.0  # r11
            left_eef[7] = 1.0  # r22
            states["left_wrist_eef"] = left_eef

            # 2. 오른손목 EEF 9D Pose (단위 회전 행렬 r11=1.0, r22=1.0 적용)
            right_eef = np.zeros(9, dtype=np.float32)
            right_eef[3] = 1.0 # r11
            right_eef[7] = 1.0 # r22
            states["right_wrist_eef"] = right_eef

            # 3. 왼손 22-DoF 손가락 관절 (라디안)
            states["left_hand_joints"] = np.zeros(22, dtype=np.float32)

            # 4. 오른손 22-DoF 손가락 관절 (라디안)
            states["right_hand_joints"] = np.zeros(22, dtype=np.float32)
        else:
            # TODO: 실제 R1 Pro Sharpa 센서로부터 실시간 상태를 읽어오세요.
            # states["left_wrist_eef"] = self.robot_client.get_left_wrist_pose_9d()
            # states["right_wrist_eef"] = self.robot_client.get_right_wrist_pose_9d()
            # states["left_hand_joints"] = self.robot_client.get_left_hand_joint_angles()
            # states["right_hand_joints"] = self.robot_client.get_right_hand_joint_angles()
            raise NotImplementedError("실제 로봇 센서/엔코더 연동 코드를 작성하세요.")

        return states

    def execute_action(self, action_step: Dict[str, np.ndarray], step_idx: int = 0):
        """
        [출력 전송] 모델이 예측한 1 타임스텝의 목표값을 R1 Pro Sharpa 모터로 전송
        
        Args:
            action_step: 각 키별 1개 타임스텝의 목표값 (shape: (D,))
            step_idx: Receding horizon 내 현재 실행 단계 번호 (0 ~ 15)
        """
        # 1. 왼손목 End-Effector 제어 목표값 추출 및 변환
        left_eef = action_step["left_wrist_eef"]
        left_pos_xyz = left_eef[:3]                     # [dx, dy, dz] 이동량 (미터)
        left_rot6d = left_eef[3:]                       # Rot6D (6차원)
        left_quat_xyzw = rot6d_to_quaternion(left_rot6d)# 쿼터니언 [qx, qy, qz, qw]
        left_euler_rpy = rot6d_to_euler(left_rot6d)     # 오일러 각도 [Roll, Pitch, Yaw] (도)

        # 2. 오른손목 End-Effector 제어 목표값 추출 및 변환
        right_eef = action_step["right_wrist_eef"]
        right_pos_xyz = right_eef[:3]
        right_rot6d = right_eef[3:]
        right_quat_xyzw = rot6d_to_quaternion(right_rot6d)
        right_euler_rpy = rot6d_to_euler(right_rot6d)

        # 3. 22-DoF 다지형 로봇손 목표 각도 추출
        left_hand_angles = action_step["left_hand_joints"]   # 22개 관절 각도 (rad)
        right_hand_angles = action_step["right_hand_joints"] # 22개 관절 각도 (rad)

        if not self.use_dummy:
            # TODO: 실제 로봇 제어기에 목표값 전송
            # self.robot_client.send_left_arm_command(pos=left_pos_xyz, quat=left_quat_xyzw)
            # self.robot_client.send_right_arm_command(pos=right_pos_xyz, quat=right_quat_xyzw)
            # self.robot_client.send_left_hand_command(angles=left_hand_angles)
            # self.robot_client.send_right_hand_command(angles=right_hand_angles)
            pass


# ==============================================================================
# Helper Functions: 파싱, 저장, 터미널 분석표 출력
# ==============================================================================
def str2bool(v: Any) -> bool:
    """True/False 문자열을 bool 타입으로 변환"""
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean 값 (True/False)이 요구됩니다.")


def parse_save_action_json(v: Any) -> Any:
    """--save-action-json 옵션 파싱 (True/False 또는 커스텀 파일 경로)"""
    if v is None:
        return True
    if isinstance(v, bool):
        return v
    v_str = str(v).strip()
    if v_str.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v_str.lower() in ("no", "false", "f", "n", "0"):
        return False
    return v_str


def save_video_frames_to_disk(
    video_dict: Dict[str, np.ndarray],
    output_dir: str,
    step_idx: int = 0,
    timestamp_str: Optional[str] = None,
) -> Dict[str, List[str]]:
    """입력 카메라 이미지들을 PNG 파일로 저장"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if timestamp_str is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = {}
    for key, val in video_dict.items():
        safe_key = key.replace("/", "_").replace(":", "_")
        saved_files[key] = []

        if isinstance(val, np.ndarray):
            frames = val[0] if val.ndim == 5 else (val if val.ndim == 4 else [val])
            for t_idx, frame in enumerate(frames):
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                img = Image.fromarray(frame)
                img_filename = f"{safe_key}_step{step_idx}_t{t_idx}_{timestamp_str}.png"
                img_path = out_path / img_filename
                img.save(img_path)
                saved_files[key].append(str(img_path.resolve()))
                print(f"[Frame Saved] Camera '{key}' (t={t_idx}) -> {img_path.resolve()}")

    return saved_files


def save_action_chunk_to_json(
    action_dict: Dict[str, np.ndarray],
    output_path: str,
    metadata: Dict[str, Any],
) -> Path:
    """예측된 Action Chunk 및 메타데이터를 구조화된 JSON 파일로 저장"""
    path = Path(output_path)
    if path.is_dir() or output_path.endswith("/") or output_path.endswith("\\"):
        path.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / f"r1_sharpa_action_{timestamp_str}.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    formatted_actions = {}
    for key, val in action_dict.items():
        if isinstance(val, np.ndarray):
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

    print(f"[JSON Saved] Action chunk successfully saved to: {path.resolve()}")
    return path


def print_r1_sharpa_action_table(action: Dict[str, np.ndarray], execution_horizon: int):
    """
    R1 Pro Sharpa 전용 예측 액션 묶음 분석표 출력
    """
    pred_steps = list(action.values())[0].shape[1]
    steps_to_exec = min(execution_horizon, pred_steps)
    
    print("\n" + "=" * 80)
    print(f"📊 [REAL_R1_PRO_SHARPA Action Output Analysis]")
    print(f" • Total Predicted Horizon : {pred_steps} timesteps (~{pred_steps/30.0:.2f}s at 30Hz)")
    print(f" • Receding Executed Steps : {steps_to_exec} timesteps (~{steps_to_exec/30.0:.2f}s)")
    print(f" • Action Modalities Breakdown:")
    print(f"   * left_wrist_eef    : Shape {str(list(action['left_wrist_eef'].shape)):<12} | 9D [Pos 3D (m) + Rot6D (6D)]")
    print(f"   * right_wrist_eef   : Shape {str(list(action['right_wrist_eef'].shape)):<12} | 9D [Pos 3D (m) + Rot6D (6D)]")
    print(f"   * left_hand_joints  : Shape {str(list(action['left_hand_joints'].shape)):<12} | 22D Hand Multi-finger Angles (rad)")
    print(f"   * right_hand_joints : Shape {str(list(action['right_hand_joints'].shape)):<12} | 22D Hand Multi-finger Angles (rad)")
    print("-" * 80)
    
    print("📍 [First Executed Step (t=0) Physical Values]:")
    # Left EEF
    l_eef = action["left_wrist_eef"][0, 0]
    l_xyz = l_eef[:3]
    l_rpy = rot6d_to_euler(l_eef[3:])
    print(f"   * Left  EEF Pos (m) : [X:{l_xyz[0]:+.4f}, Y:{l_xyz[1]:+.4f}, Z:{l_xyz[2]:+.4f}] | Rot: [R:{l_rpy[0]:+.1f}°, P:{l_rpy[1]:+.1f}°, Y:{l_rpy[2]:+.1f}°]")
    
    # Right EEF
    r_eef = action["right_wrist_eef"][0, 0]
    r_xyz = r_eef[:3]
    r_rpy = rot6d_to_euler(r_eef[3:])
    print(f"   * Right EEF Pos (m) : [X:{r_xyz[0]:+.4f}, Y:{r_xyz[1]:+.4f}, Z:{r_xyz[2]:+.4f}] | Rot: [R:{r_rpy[0]:+.1f}°, P:{r_rpy[1]:+.1f}°, Y:{r_rpy[2]:+.1f}°]")
    
    # Left Hand 22D
    l_hand = action["left_hand_joints"][0, 0]
    l_hand_sample = ", ".join([f"{x:+.3f}" for x in l_hand[:5]])
    print(f"   * Left  Hand Joints : [{l_hand_sample}, ... (total 22 DoF rad)]")

    # Right Hand 22D
    r_hand = action["right_hand_joints"][0, 0]
    r_hand_sample = ", ".join([f"{x:+.3f}" for x in r_hand[:5]])
    print(f"   * Right Hand Joints : [{r_hand_sample}, ... (total 22 DoF rad)]")
    print("=" * 80 + "\n")


# ==============================================================================
# Main Control Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="REAL_R1_PRO_SHARPA Dedicated Zero-Shot Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # --------------------------------------------------------------------------
    # [사용자 설정 파라미터]
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--model-path",
        type=str,
        default="nvidia/GR00T-N1.7-3B",
        help="HuggingFace 모델 ID 또는 로컬 가중치 디렉토리 경로",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="추론 디바이스 (cuda:0 또는 cpu)",
    )
    parser.add_argument(
        "--task-instruction",
        type=str,
        default="Pick up the red mug with left arm and place it on the right table",
        help="R1 Pro Sharpa 로봇에게 내릴 자연어 작업 지시문",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=16,
        help="1회 추론 시 생성된 40개 스텝 중 실제 실행할 단계 수 (권장: 16)",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=1,
        help="실행할 제어 사이클 반복 횟수 (테스트용)",
    )
    parser.add_argument(
        "--save-action-json",
        type=parse_save_action_json,
        nargs="?",
        const=True,
        default=False,
        help="예측된 Action Chunk를 JSON 파일로 저장할지 여부 (True/False)",
    )
    parser.add_argument(
        "--save-action-json-path",
        type=str,
        default="output_actions/r1_sharpa_actions.json",
        help="Action JSON 파일 저장 경로",
    )
    parser.add_argument(
        "--save-video-frames",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="모델 입력으로 사용된 3개 카메라 이미지를 PNG 파일로 저장할지 여부 (True/False)",
    )
    parser.add_argument(
        "--save-frames-dir",
        type=str,
        default="output_frames/r1_sharpa",
        help="카메라 이미지가 저장될 디렉토리 경로",
    )

    args = parser.parse_args()

    # JSON 저장 경로 해석
    if isinstance(args.save_action_json, str):
        save_json_enabled = True
        json_save_path = args.save_action_json
    elif isinstance(args.save_action_json, bool):
        save_json_enabled = args.save_action_json
        json_save_path = args.save_action_json_path if save_json_enabled else None
    else:
        save_json_enabled = False
        json_save_path = None

    tag = EmbodimentTag.REAL_R1_PRO_SHARPA

    print("\n" + "=" * 80)
    print("🤖 [REAL_R1_PRO_SHARPA Zero-Shot Policy Initializing]")
    print(f" • Model Path         : {args.model_path}")
    print(f" • Embodiment Tag     : {tag.name} ('{tag.value}')")
    print(f" • Inference Device   : {args.device}")
    print(f" • Task Instruction   : '{args.task_instruction}'")
    print(f" • Save Action JSON   : {save_json_enabled}" + (f" -> {json_save_path}" if save_json_enabled else ""))
    print(f" • Save Video Frames  : {args.save_video_frames}" + (f" -> {args.save_frames_dir}" if args.save_video_frames else ""))
    print("=" * 80 + "\n")

    # 1. GR00T Policy 모델 로드 (REAL_R1_PRO_SHARPA 전용 태그 바인딩)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=tag,
        device=args.device,
        strict=True,
    )

    # 2. R1 Pro Sharpa 로봇 하드웨어 인터페이스 초기화
    robot = R1ProSharpaRobot(use_dummy=True)

    # 3. 제어 루프 파라미터 (R1 Pro Sharpa는 Temporal Horizon T=2 요구)
    v_horizon = 2  # delta_indices: [-20, 0] (과거 1프레임 + 현재 1프레임 시퀀스)
    s_horizon = 1  # delta_indices: [0]
    l_horizon = 1

    task_prompt = args.task_instruction
    execution_horizon = args.execution_horizon

    # 4. 제어 루프 실행
    try:
        for cycle_idx in range(args.num_cycles):
            print(f"\n▶ [Cycle {cycle_idx + 1}/{args.num_cycles}] Collecting Observations from R1 Pro Sharpa Sensors...")
            t_start = time.time()

            # ------------------------------------------------------------------
            # Step A: 3개 카메라 및 관절 센서로부터 Observation 수집
            # ------------------------------------------------------------------
            camera_dict = robot.capture_cameras()
            state_dict = robot.get_robot_states()

            # 비디오 입력 텐서 구성: Shape (B=1, T=2, H=240, W=320, C=3)
            video_input = {}
            for v_key in robot.VIDEO_KEYS:
                raw_img = camera_dict[v_key]
                # T=2 타임스텝 시퀀스 형성
                img_sequence = np.stack([raw_img] * v_horizon, axis=0)
                video_input[v_key] = np.expand_dims(img_sequence, axis=0)

            # 상태 입력 텐서 구성: Shape (B=1, T=1, D)
            state_input = {}
            for s_key in robot.STATE_KEYS:
                raw_state = state_dict[s_key]
                state_sequence = np.stack([raw_state] * s_horizon, axis=0)
                state_input[s_key] = np.expand_dims(state_sequence, axis=0).astype(np.float32)

            # 언어 지시문 구성 (R1 Pro Sharpa 규격: 'annotation.human.coarse_action')
            req_lang_keys = policy.get_modality_config()["language"].modality_keys
            lang_key = req_lang_keys[0] if req_lang_keys else "annotation.human.coarse_action"
            language_input = {
                lang_key: [[task_prompt] * l_horizon]
            }

            observation = {
                "video": video_input,
                "state": state_input,
                "language": language_input,
            }

            # 선택 사항: 카메라 프레임 디스크 저장
            saved_frames_info = {}
            if args.save_video_frames:
                saved_frames_info = save_video_frames_to_disk(
                    video_dict=video_input,
                    output_dir=args.save_frames_dir,
                    step_idx=cycle_idx,
                )

            # ------------------------------------------------------------------
            # Step B: 모델 정책 추론 (Action Chunk 40단계 생성)
            # ------------------------------------------------------------------
            print(f"⏳ Running DiT Diffusion Policy Inference on {args.device}...")
            action, info = policy.get_action(observation)
            t_infer = time.time() - t_start
            print(f"✅ Inference Complete! Time elapsed: {t_infer*1000:.1f} ms")

            # ------------------------------------------------------------------
            # Step C: R1 Pro Sharpa 예측 액션 분석표 출력
            # ------------------------------------------------------------------
            print_r1_sharpa_action_table(action, execution_horizon)

            # 선택 사항: 액션 JSON 저장
            if save_json_enabled and json_save_path:
                action_shapes = {k: list(v.shape) for k, v in action.items()}
                metadata = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model_path": args.model_path,
                    "embodiment_tag": tag.name,
                    "embodiment_value": tag.value,
                    "task_instruction": task_prompt,
                    "inference_time_ms": round(t_infer * 1000, 2),
                    "action_horizon": 40,
                    "execution_horizon": execution_horizon,
                    "action_shapes": action_shapes,
                }
                if saved_frames_info:
                    metadata["saved_video_frames"] = saved_frames_info
                save_action_chunk_to_json(action, json_save_path, metadata)

            # ------------------------------------------------------------------
            # Step D: Receding Horizon 모터 실행
            # ------------------------------------------------------------------
            pred_steps = list(action.values())[0].shape[1]
            steps_to_exec = min(execution_horizon, pred_steps)
            print(f"🚀 Sending first {steps_to_exec} steps to R1 Pro Sharpa motors at 30Hz...")

            for step_idx in range(steps_to_exec):
                current_action_step = {
                    "left_wrist_eef": action["left_wrist_eef"][0, step_idx, :],
                    "right_wrist_eef": action["right_wrist_eef"][0, step_idx, :],
                    "left_hand_joints": action["left_hand_joints"][0, step_idx, :],
                    "right_hand_joints": action["right_hand_joints"][0, step_idx, :],
                }

                robot.execute_action(current_action_step, step_idx=step_idx)
                time.sleep(1.0 / 30.0)  # 30 FPS 로봇 실행 레이트

    except KeyboardInterrupt:
        print("\n[Control Loop] Stopped by user.")


if __name__ == "__main__":
    main()
