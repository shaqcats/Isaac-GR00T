# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
================================================================================
Isaac GR00T N1.7 Dual-Arm Robot Zero-Shot Inference Script
================================================================================

본 스크립트는 NVIDIA의 사전학습 VLA 파운데이션 모델 `nvidia/GR00T-N1.7-3B`를 활용하여
사용자 정의 양팔 로봇(Dual-Arm Robot / Humanoid)에 Zero-Shot 정책을 적용하는 표준 템플릿입니다.

--------------------------------------------------------------------------------
[1] 하드웨어 실행 환경 설정 (DGX Spark / aarch64 / PC 공통)
--------------------------------------------------------------------------------
    * 주의: DGX Spark 환경에서는 가상환경 패키지 충돌 방지를 위해 `uv run` 대신
      직접 activate 스크립트를 실행한 후 python을 실행해야 합니다.

    $ source .venv/bin/activate
    $ source scripts/activate_spark.sh  (DGX Spark 환경인 경우)
    $ python scripts/zero_shot_dual_arm_inference.py --help

--------------------------------------------------------------------------------
[2] 주요 실행 예시
--------------------------------------------------------------------------------
    # 1. R1 Pro Sharpa 양팔 로봇 제로샷 추론 (기본값)
    $ python scripts/zero_shot_dual_arm_inference.py \
          --embodiment-tag REAL_R1_PRO_SHARPA \
          --task-instruction "Pick up the red mug with left arm and place it on the right table"

    # 2. Unitree G1 휴머노이드 양팔 제로샷 추론
    $ python scripts/zero_shot_dual_arm_inference.py \
          --embodiment-tag REAL_G1 \
          --task-instruction "Wave both hands hello"

    # 3. 모델 입력 카메라 이미지 및 예측 액션 JSON 동시 저장
    $ python scripts/zero_shot_dual_arm_inference.py \
          --embodiment-tag REAL_R1_PRO_SHARPA \
          --save-action-json True \
          --save-video-frames True \
          --save-frames-dir output_frames/my_experiment
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
# [USER GUIDE] 1. 입력 데이터(Observation) 및 출력 데이터(Action) 상세 규격 설명
# ==============================================================================
"""
================================================================================
📥 1. 입력 데이터 (Model Inputs / Observations) 상세 설명
================================================================================
GR00T 정책 모델은 3가지 모달리티(Video, State, Language) 입력을 받습니다.

1) Video (카메라 영상)
   - 구조: Dict[str, np.ndarray], shape: (Batch=1, Horizon=T, Height, Width, Channels=3)
   - 데이터 타입: uint8, RGB 색상 채널, 픽셀 범위 [0, 255]
   - Temporal Horizon (T):
     * REAL_R1_PRO_SHARPA: T=2 (과거 1프레임 + 현재 1프레임, delta_indices=[-20, 0])
     * REAL_G1: T=1 (현재 1프레임, delta_indices=[0])
   - Embodiment별 요구 카메라 키:
     * REAL_R1_PRO_SHARPA:
       - 'ego_view_res320x240_freq20'        : 로봇 머리/가슴 Ego 시점 (320x240)
       - 'left_wrist_view_res320x240_freq20' : 왼손 손목 카메라 (320x240)
       - 'right_wrist_view_res320x240_freq20': 오른손 손목 카메라 (320x240)
     * REAL_G1:
       - 'ego_view'                          : 로봇 머리 Ego 시점 (224x224)

2) State (로봇 현재 관절 / EEF 상태)
   - 구조: Dict[str, np.ndarray], shape: (Batch=1, Horizon=1, Dimension=D)
   - 데이터 타입: float32
   - Embodiment별 요구 상태 키 & 차원(D):
     * REAL_R1_PRO_SHARPA:
       - 'left_wrist_eef'   : 9D [x, y, z, rot6d(6차원)] - 왼손목 End-Effector Pose
       - 'right_wrist_eef'  : 9D [x, y, z, rot6d(6차원)] - 오른손목 End-Effector Pose
       - 'left_hand_joints' : 22D - 왼손 22-DoF 다지형 그리퍼 관절 각도 (라디안)
       - 'right_hand_joints': 22D - 오른손 22-DoF 다지형 그리퍼 관절 각도 (라디안)
     * REAL_G1:
       - 'left_wrist_eef_9d', 'right_wrist_eef_9d' : 9D EEF Pose
       - 'left_arm', 'right_arm'                   : 7D 팔 관절 각도 (라디안)
       - 'left_hand', 'right_hand'                 : 7D 손가락 관절 각도 (라디안)
       - 'waist'                                   : 3D 허리 관절 각도 (라디안)

3) Language (자연어 작업 지시문)
   - 구조: Dict[str, List[List[str]]], 예: {"instruction": [["작업 지시문"]]}

================================================================================
📤 2. 출력 데이터 (Model Outputs / Action Trajectory) 상세 설명
================================================================================
모델 추론 결과 `action = policy.get_action(observation)[0]`은
미래 40개 타임스텝에 대한 **Receding Action Chunk (궤적)**를 반환합니다.

- 구조: Dict[str, np.ndarray], shape: (Batch=1, Horizon=40, Dimension=D)
- 제어 주기: ~30 Hz (각 타임스텝 간격 Δt ≈ 33.3 ms, 40단계 ≈ 1.33초 동안의 미래 궤적)
- Receding Horizon 실행: 모델이 생성한 40개 타임스텝 중 첫 16개 단계(`execution_horizon=16`)를
  로봇 모터에 순차 전송한 후, 새로운 카메라/관절 상태를 다시 캡처하여 다음 추론을 수행합니다.

- Action 키별 내부 값(D) 해석:
  ──────────────────────────────────────────────────────────────────────────────
  A. 9차원 End-Effector Pose ('left_wrist_eef', 'right_wrist_eef')
     * index 0 ~ 2 (3D): 직교 좌표계 목표 이동량 [dx, dy, dz] (단위: 미터 [m])
     * index 3 ~ 8 (6D): 목표 회전 성분 Rot6D [r11, r21, r31, r12, r22, r32]
       -> 3x3 회전 행렬의 첫 2개 열(column) 벡터를 의미합니다.
       -> 본 스크립트의 `rot6d_to_quaternion()` 함수로 Quaternion [x, y, z, w] 또는
          Euler 각도 [roll, pitch, yaw]로 손쉽게 변환하여 모터/IK 제어기에 전달할 수 있습니다.

  B. 22차원 다지형 손 관절 ('left_hand_joints', 'right_hand_joints')
     * index 0 ~ 21 (22D): 엄지, 검지, 중지, 약지, 소지 및 손바닥 관절의 목표 회전 각도 (단위: 라디안 [rad])

  C. 7차원 팔 관절 각도 ('left_arm', 'right_arm')
     * index 0 ~ 6 (7D): 어깨(3-DoF), 팔꿈치(2-DoF), 손목(2-DoF) 관절의 목표 각도 (단위: 라디안 [rad])
  ──────────────────────────────────────────────────────────────────────────────
"""


# ==============================================================================
# [MATH UTILS] Rot6D 변환 유틸리티 (9D EEF Pose 해석용)
# ==============================================================================
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Rot6D 6차원 벡터 [r11, r21, r31, r12, r22, r32]를 Gram-Schmidt 정규화를 거쳐
    3x3 직교 회전 행렬(Rotation Matrix)로 복원합니다.
    """
    col1 = rot6d[:3]
    col2 = rot6d[3:6]
    
    # 1. First column normalization
    c1_norm = np.linalg.norm(col1)
    if c1_norm < 1e-6:
        col1 = np.array([1.0, 0.0, 0.0])
    else:
        col1 = col1 / c1_norm
        
    # 2. Second column orthogonalization & normalization
    col2 = col2 - np.dot(col1, col2) * col1
    c2_norm = np.linalg.norm(col2)
    if c2_norm < 1e-6:
        col2 = np.array([0.0, 1.0, 0.0])
    else:
        col2 = col2 / c2_norm
        
    # 3. Third column is cross product
    col3 = np.cross(col1, col2)
    
    # Assemble 3x3 rotation matrix
    rot_matrix = np.column_stack([col1, col2, col3])
    return rot_matrix


def rot6d_to_quaternion(rot6d: np.ndarray) -> np.ndarray:
    """
    Rot6D 6차원 벡터를 쿼터니언 [qx, qy, qz, qw] (SciPy 규격)으로 변환합니다.
    """
    matrix = rot6d_to_matrix(rot6d)
    return Rotation.from_matrix(matrix).as_quat()


def rot6d_to_euler(rot6d: np.ndarray, seq: str = "xyz", degrees: bool = True) -> np.ndarray:
    """
    Rot6D 6차원 벡터를 오일러 각도 (기본: roll, pitch, yaw [도 단위])로 변환합니다.
    """
    matrix = rot6d_to_matrix(rot6d)
    return Rotation.from_matrix(matrix).as_euler(seq, degrees=degrees)


# ==============================================================================
# [USER CONFIG 2] Real Robot HW Interface (사용자 하드웨어 연동 클래스)
# ==============================================================================
class CustomDualArmRobot:
    """
    [사용자 작성 영역] 실제 로봇 하드웨어 SDK / 드라이버 연동 인터페이스
    
    실제 로봇에 연결할 때는 이 클래스 내부의 TODO 부분을 로봇 SDK
    (예: Unitree SDK, Franka Control, ROS2 Node, RealSense SDK, OpenCV 등)로 대체하세요.
    """

    def __init__(self, use_dummy: bool = True):
        self.use_dummy = use_dummy
        print(f"[RobotInterface] Initializing Dual-Arm Interface (Dummy Mode: {self.use_dummy})...")
        
        if not self.use_dummy:
            # TODO: 실제 카메라 드라이버 초기화 (예: pyrealsense2, cv2.VideoCapture 등)
            # TODO: 실제 로봇 팔 / 손 모터 통신 초기화 (예: ROS2 Subscriber/Publisher, CAN 통신 등)
            pass

    def capture_cameras(self, video_keys: List[str]) -> Dict[str, np.ndarray]:
        """
        [USER INPUT 1] 카메라 영상 획득 인터페이스
        
        Args:
            video_keys: 모델이 요구하는 카메라 키 목록
                        (예: ['ego_view_res320x240_freq20', 'left_wrist_view_res320x240_freq20', ...])
        Returns:
            Dict[str, np.ndarray]: 키별 (Height, Width, 3) 크기의 RGB uint8 이미지 [0 ~ 255]
        """
        cameras = {}
        for key in video_keys:
            # 키 이름에 '320x240'이 포함되어 있으면 240x320, 기본값은 224x224
            if "320x240" in key:
                h, w = 240, 320
            else:
                h, w = 224, 224

            if self.use_dummy:
                # 더미 난수 RGB 이미지 생성
                cameras[key] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            else:
                # TODO: 실제 카메라에서 최신 프레임을 읽어와 반환하세요.
                # frame = my_camera_driver.get_frame(key) # RGB uint8 (H, W, 3)
                # cameras[key] = frame
                raise NotImplementedError("실제 카메라 SDK 연동 코드를 구현하세요.")

        return cameras

    def get_joint_states(self, state_keys: List[str], state_dims: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        """
        [USER INPUT 2] 로봇 현재 관절 각도 및 End-Effector 상태 획득 인터페이스
        
        Args:
            state_keys: 모델이 요구하는 상태 키 목록
            state_dims: 각 키별 요구 차원(D) 딕셔너리
        Returns:
            Dict[str, np.ndarray]: 키별 (D,) 크기의 float32 관절/상태 벡터
        """
        states = {}
        state_dims = state_dims or {}
        
        for key in state_keys:
            # 키에 맞는 기본 차원(D) 결정
            if key in state_dims:
                dim = state_dims[key]
            elif "wrist_eef" in key:
                dim = 9   # EEF Pose (x, y, z, r11, r21, r31, r12, r22, r32)
            elif "hand_joints" in key:
                dim = 22  # R1 Pro Sharpa 22-DoF Dexterous Hand
            elif "arm" in key or "hand" in key:
                dim = 7   # 7-DoF Arm / 7-DoF Hand
            elif "leg" in key:
                dim = 6   # 6-DoF Leg
            elif "waist" in key:
                dim = 3   # 3-DoF Torso
            else:
                dim = 7

            if self.use_dummy:
                if "wrist_eef" in key:
                    # 9D Pose 더미 생성 (특이점 방지를 위한 단위 회전 행렬 rot6d r11=1.0, r22=1.0 적용)
                    st = np.zeros(dim, dtype=np.float32)
                    st[3] = 1.0  # r11 = 1.0
                    st[7] = 1.0  # r22 = 1.0
                    states[key] = st
                else:
                    states[key] = np.zeros(dim, dtype=np.float32)
            else:
                # TODO: 실제 로봇 엔코더 / 센서에서 현재 상태를 읽어와 반환하세요.
                # states[key] = my_robot_driver.get_joint_positions(key) # np.float32 shape (D,)
                raise NotImplementedError("실제 로봇 관절 센서 연동 코드를 구현하세요.")

        return states

    def execute_action(self, action_step: Dict[str, np.ndarray], step_idx: int = 0):
        """
        [USER OUTPUT HANDLER] 모델이 예측한 1-Step 액션 명령을 실제 로봇 모터로 전송
        
        Args:
            action_step: 각 키별 1 타임스텝의 제어 목표값 (shape: (D,))
            step_idx: Receding horizon 내 현재 실행 단계 번호 (0 ~ 15)
        """
        # ======================================================================
        # [출력값 해석 및 하드웨어 모터 전송 예시]
        # ======================================================================
        if "left_wrist_eef" in action_step:
            eef_9d = action_step["left_wrist_eef"]
            trans_xyz = eef_9d[:3]                 # 이동량 [dx, dy, dz] (미터)
            rot6d = eef_9d[3:]                     # 회전 Rot6D (6차원)
            quat_xyzw = rot6d_to_quaternion(rot6d) # 쿼터니언 [qx, qy, qz, qw]
            euler_rpy = rot6d_to_euler(rot6d)      # 오일러 각도 [roll, pitch, yaw] (도)
            
            # 실제 로봇 IK(Inverse Kinematics) 또는 카테시안 임피던스 제어기에 목표값 전달:
            # my_robot_driver.send_cartesian_target(arm="left", pos=trans_xyz, quat=quat_xyzw)

        if "left_hand_joints" in action_step:
            hand_22d = action_step["left_hand_joints"] # 22-DoF 각도 (라디안)
            # my_robot_driver.send_hand_joint_targets(hand="left", angles=hand_22d)

        if "left_arm" in action_step:
            arm_7d = action_step["left_arm"] # 7-DoF 팔 관절 각도 (라디안)
            # my_robot_driver.send_arm_joint_targets(arm="left", angles=arm_7d)


# ==============================================================================
# Helper Functions: 입출력 데이터 저장 및 파싱
# ==============================================================================
def str2bool(v: Any) -> bool:
    """True/False 문자열을 파이썬 bool 타입으로 변환"""
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
    """모델 입력으로 들어간 비디오 카메라 프레임들을 PNG 파일로 저장"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if timestamp_str is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = {}
    for key, val in video_dict.items():
        safe_key = key.replace("/", "_").replace(":", "_")
        saved_files[key] = []

        if isinstance(val, np.ndarray):
            if val.ndim == 5:
                frames = val[0]
            elif val.ndim == 4:
                frames = val
            elif val.ndim == 3:
                frames = [val]
            else:
                continue

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
    """예측된 40단계 Action Chunk 및 메타데이터를 구조화된 JSON 파일로 저장"""
    path = Path(output_path)
    if path.is_dir() or output_path.endswith("/") or output_path.endswith("\\"):
        path.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / f"action_chunk_{timestamp_str}.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    formatted_actions = {}
    for key, val in action_dict.items():
        if isinstance(val, np.ndarray):
            # Batch size가 1인 경우 가독성을 위해 (T, D) 형태로 저장
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


def print_action_summary_table(action: Dict[str, np.ndarray], execution_horizon: int):
    """
    콘솔 터미널에 모델 추론 결과(Action Chunk)의 상세 구조 및 첫 스텝 제어값을 보기 쉽게 출력
    """
    pred_steps = list(action.values())[0].shape[1]
    steps_to_exec = min(execution_horizon, pred_steps)
    
    print("\n" + "=" * 78)
    print(f"📊 [Action Chunk Output Analysis]")
    print(f" - Total Predicted Horizon : {pred_steps} timesteps (~{pred_steps/30.0:.2f} sec at 30Hz)")
    print(f" - Receding Executed Steps : {steps_to_exec} timesteps (~{steps_to_exec/30.0:.2f} sec)")
    print(f" - Output Modalities Breakdown:")
    
    for a_key, a_val in action.items():
        dim = a_val.shape[-1]
        desc = ""
        if "wrist_eef" in a_key:
            desc = "9D Cartesian EEF Pose [Translation 3D (m) + Rot6D (6D)]"
        elif "hand_joints" in a_key:
            desc = f"{dim}D Dexterous Hand Multi-finger Joint Angles (rad)"
        elif "arm" in a_key:
            desc = f"{dim}D Arm Joint Positions (rad)"
        elif "hand" in a_key:
            desc = f"{dim}D Gripper / Hand Joint Positions (rad)"
        elif "waist" in a_key:
            desc = "3D Torso / Waist Joint Positions (rad)"
        else:
            desc = f"{dim}D Control Command"
            
        print(f"   * {a_key:<20} : Shape {str(list(a_val.shape)):<12} | {desc}")
        
    print("-" * 78)
    print("📍 [First Executed Step (t=0) Sample Values]:")
    for a_key, a_val in action.items():
        step0_val = a_val[0, 0, :]
        if "wrist_eef" in a_key:
            xyz = step0_val[:3]
            rpy = rot6d_to_euler(step0_val[3:])
            print(f"   * {a_key:<20} -> Pos(m): [X:{xyz[0]:+.4f}, Y:{xyz[1]:+.4f}, Z:{xyz[2]:+.4f}] | Rot(deg): [R:{rpy[0]:+.1f}°, P:{rpy[1]:+.1f}°, Y:{rpy[2]:+.1f}°]")
        elif len(step0_val) <= 7:
            val_str = ", ".join([f"{x:+.3f}" for x in step0_val])
            print(f"   * {a_key:<20} -> [{val_str}]")
        else:
            val_str = ", ".join([f"{x:+.3f}" for x in step0_val[:5]])
            print(f"   * {a_key:<20} -> [{val_str}, ... (total {len(step0_val)} dim)]")
    print("=" * 78 + "\n")


# ==============================================================================
# Main Control Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="GR00T N1.7 Zero-Shot Dual-Arm Policy Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # --------------------------------------------------------------------------
    # [사용자 입력 1] 모델 및 로봇 태그 설정
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--model-path",
        type=str,
        default="nvidia/GR00T-N1.7-3B",
        help="HuggingFace 모델 ID 또는 로컬 가중치 디렉토리 경로",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="REAL_R1_PRO_SHARPA",
        help="로봇 Embodiment 태그 (REAL_R1_PRO_SHARPA, REAL_G1, XDOF 등)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="추론 연산 디바이스 (cuda:0 또는 cpu)",
    )
    
    # --------------------------------------------------------------------------
    # [사용자 입력 2] 작업 지시문 및 제어 주기 설정
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--task-instruction",
        type=str,
        default="Pick up the red mug with left arm and place it on the right table",
        help="로봇이 수행할 자연어 작업 프롬프트 명령어",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=16,
        help="1회 추론 시 예측된 40단계 중 실제 실행할 타임스텝 수 (<= 40)",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=1,
        help="실행할 제어 루프 반복 횟수 (테스트용, 무한루프는 while True 활용)",
    )

    # --------------------------------------------------------------------------
    # [사용자 입력 3] 입출력 파일 저장 옵션
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--save-action-json",
        type=parse_save_action_json,
        nargs="?",
        const=True,
        default=False,
        help="예측된 Action Chunk를 JSON 파일로 저장할지 여부 (True/False 또는 커스텀 파일 경로)",
    )
    parser.add_argument(
        "--save-action-json-path",
        type=str,
        default="output_actions/predicted_actions.json",
        help="--save-action-json True일 때 저장할 기본 JSON 파일 경로",
    )
    parser.add_argument(
        "--save-video-frames",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="모델 입력으로 사용된 카메라 영상 프레임들을 PNG 파일로 저장할지 여부 (True/False)",
    )
    parser.add_argument(
        "--save-frames-dir",
        type=str,
        default="output_frames",
        help="--save-video-frames True일 때 이미지가 저장될 디렉토리 경로",
    )

    args = parser.parse_args()

    # JSON 저장 설정 파싱
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

    print("\n" + "=" * 78)
    print("🤖 [GR00T Zero-Shot Policy Initializing]")
    print(f" - Model Checkpoint   : {args.model_path}")
    print(f" - Embodiment Tag     : {tag.name} (Modality config: '{tag.value}')")
    print(f" - Device             : {args.device}")
    print(f" - Task Instruction   : '{args.task_instruction}'")
    print(f" - Save Action JSON   : {save_json_enabled}" + (f" -> {json_save_path}" if save_json_enabled else ""))
    print(f" - Save Video Frames  : {args.save_video_frames}" + (f" -> {args.save_frames_dir}" if args.save_video_frames else ""))
    print("=" * 78 + "\n")

    # 1. 모델 정책 로드
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=tag,
        device=args.device,
        strict=True,
    )

    # 2. 모델의 요구 모달리티 규격 조회
    modality_configs = policy.get_modality_config()
    req_video_keys = modality_configs["video"].modality_keys
    req_state_keys = modality_configs["state"].modality_keys
    req_lang_keys = modality_configs["language"].modality_keys

    v_horizon = len(modality_configs["video"].delta_indices)  # 예: R1 Sharpa=2, G1=1
    s_horizon = len(modality_configs["state"].delta_indices)  # 예: 1
    l_horizon = len(modality_configs["language"].delta_indices)

    # 3. 로봇 하드웨어 인터페이스 초기화 (더미 모드)
    robot = CustomDualArmRobot(use_dummy=True)

    task_prompt = args.task_instruction
    execution_horizon = args.execution_horizon

    # 4. 폐루프 제어 루프 실행
    try:
        for cycle_idx in range(args.num_cycles):
            print(f"\n▶ [Control Cycle {cycle_idx + 1}/{args.num_cycles}] Collecting Observations...")
            t_start = time.time()

            # ------------------------------------------------------------------
            # Step A: 로봇 센서로부터 입력값(Observation) 획득
            # ------------------------------------------------------------------
            camera_dict = robot.capture_cameras(req_video_keys)
            state_dict = robot.get_joint_states(req_state_keys)

            # 비디오 입력 텐서 구성: Shape (B=1, T=v_horizon, H, W, C=3)
            video_input = {}
            for v_key in req_video_keys:
                raw_img = camera_dict[v_key]
                img_sequence = np.stack([raw_img] * v_horizon, axis=0)
                video_input[v_key] = np.expand_dims(img_sequence, axis=0)

            # 관절/상태 입력 텐서 구성: Shape (B=1, T=s_horizon, D)
            state_input = {}
            for s_key in req_state_keys:
                raw_state = state_dict[s_key]
                state_sequence = np.stack([raw_state] * s_horizon, axis=0)
                state_input[s_key] = np.expand_dims(state_sequence, axis=0).astype(np.float32)

            # 언어 지시문 입력 구성
            language_input = {
                req_lang_keys[0]: [[task_prompt] * l_horizon]
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
            # Step B: 모델 정책 추론 (Action Chunk 예측)
            # ------------------------------------------------------------------
            print(f"⏳ Running DiT Diffusion Policy Inference on {args.device}...")
            action, info = policy.get_action(observation)
            t_infer = time.time() - t_start
            print(f"✅ Inference Complete! Time elapsed: {t_infer*1000:.1f} ms")

            # ------------------------------------------------------------------
            # Step C: 출력값(Action Chunk) 구조 상세 출력
            # ------------------------------------------------------------------
            print_action_summary_table(action, execution_horizon)

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
                    "action_horizon": list(action.values())[0].shape[1],
                    "execution_horizon": execution_horizon,
                    "action_shapes": action_shapes,
                }
                if saved_frames_info:
                    metadata["saved_video_frames"] = saved_frames_info
                save_action_chunk_to_json(action, json_save_path, metadata)

            # ------------------------------------------------------------------
            # Step D: Receding Horizon 순차 실행
            # ------------------------------------------------------------------
            pred_steps = list(action.values())[0].shape[1]
            steps_to_exec = min(execution_horizon, pred_steps)
            print(f"🚀 Executing first {steps_to_exec} steps on robot hardware at 30Hz...")

            for step_idx in range(steps_to_exec):
                current_action_step = {}
                for a_key, a_val in action.items():
                    current_action_step[a_key] = a_val[0, step_idx, :]  # Shape: (D,)

                robot.execute_action(current_action_step, step_idx=step_idx)
                time.sleep(1.0 / 30.0)  # 30 FPS 로봇 제어 주기

    except KeyboardInterrupt:
        print("\n[Control Loop] Stopped by user.")


if __name__ == "__main__":
    main()
