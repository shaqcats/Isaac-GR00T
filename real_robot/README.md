# Isaac GR00T Real Robot Deployment & Real-Time Control Guide

본 가이드는 **Isaac GR00T N1.7 VLA(Vision-Language-Action) 모델을 실제 물리 양팔/휴머노이드 로봇에 안전하고 실시간(Real-Time)으로 적용**하기 위한 개발자 사용자 가이드입니다.

---

## 1. 아키텍처 개요 (Real-Time Architecture)

GR00T VLA 모델 추론(비전 인코더 + DiT 디퓨전 디노이징)은 모델 크기상 약 15~40ms의 시간이 소요됩니다. 물리 로봇 모터의 30~100Hz 주기 제어와 끊김없는 상호작용을 보장하기 위해 **비동기 멀티스레딩 파이프라인**으로 설계되었습니다.

```mermaid
graph TD
    subgraph Sensing["Sensing Layer (Async Capture Thread)"]
        Cam1[Ego Camera V4L2/RealSense] --> RB[Thread-Safe Ring Buffer]
        Cam2[Left Wrist Camera] --> RB
        Cam3[Right Wrist Camera] --> RB
        HWState[Joint/EEF Hardware Feedback] --> StateCache[State Cache]
    end

    subgraph Inference["Policy Inference Layer (Thread 1)"]
        RB --> Observation[Observation Builder]
        StateCache --> Observation
        Prompt[Task Instruction] --> Observation
        Observation --> Policy[Gr00tPolicy N1.7-3B]
        Policy --> ActionChunk["Predicted Action Chunk (1, 40, D)"]
    end

    subgraph Control["Real-Time Control Layer (Thread 2: 30-100Hz)"]
        ActionChunk --> Queue[Receding Horizon Action Buffer]
        Queue --> Recede[Action Step Extractor]
        Recede --> Safety[Safety Guard & Clamping]
        Safety --> Smooth[EMA Velocity Smoothing]
        Smooth --> HardwareCmd[robot.send_action()]
    end

    HardwareCmd --> Motors[Physical Robot Motors]
```

---

## 2. 모듈 디렉토리 구성

사용자가 작업할 수 있도록 `real_robot/` 공간이 모듈화되어 구성되어 있습니다:

```text
Isaac-GR00T/
├── real_robot/
│   ├── __init__.py
│   ├── robot_interface.py     # ⭐ [사용자 수정 1] 실제 로봇 SDK (모터 제어/관절 피드백) 연동
│   ├── camera_interface.py    # ⭐ [사용자 수정 2] 실제 카메라 (OpenCV, RealSense) 드라이버 연동
│   ├── safety_guard.py        # 하드웨어 안전 보호 장치 (관절/속도/작업공간 Clamping, E-Stop)
│   ├── async_controller.py    # 비동기 추론 & 실시간 모터 제어 루프
│   ├── run_real_robot.py      # 실시간 실행 CLI 엔트리포인트
│   └── README.md              # 본 가이드 문서
```

---

## 3. 사용자 로봇 SDK 연동 (4단계)

[`real_robot/robot_interface.py`](robot_interface.py) 파일의 `CustomUserDualArmRobot` 클래스를 상속받거나 수정하여 실제 로봇의 라이브러리를 연결합니다.

```python
from real_robot.robot_interface import BaseRobotHardware

class MyDualArmRobot(BaseRobotHardware):
    def __init__(self, robot_ip="192.168.123.10"):
        super().__init__(name="MyDualArmRobot")
        self.robot_ip = robot_ip
        self.sdk = None

    # [Step 1] 모터 및 통신 버스 연결
    def connect(self) -> bool:
        # 예: self.sdk = MyVendorSDK.connect(self.robot_ip)
        # self.sdk.enable_actuators()
        self._connected = True
        return True

    # [Step 2] 안전 종료 및 모터 해제
    def disconnect(self):
        # self.sdk.disable_actuators()
        self._connected = False

    # [Step 3] 현재 관절 각도 / EEF Pose 센서값 읽기
    def get_state(self, state_keys: list[str], state_dims: dict = None) -> dict[str, np.ndarray]:
        states = {}
        # 예: left_q = self.sdk.get_left_arm_joint_positions() # (7,)
        # 예: left_eef = self.sdk.get_left_eef_pose()          # 9D [x,y,z, rot6d]
        for key in state_keys:
            if "wrist_eef" in key:
                st = np.zeros(9, dtype=np.float32)
                st[3] = 1.0; st[7] = 1.0  # Rot6D 단위 회전 설정 필수
                states[key] = st
            else:
                states[key] = np.zeros(state_dims.get(key, 7), dtype=np.float32)
        return states

    # [Step 4] 모델이 계산한 단일 제어 스텝 액션 모터에 전송
    def send_action(self, action_step: dict[str, np.ndarray]) -> bool:
        # if "left_wrist_eef" in action_step:
        #     self.sdk.set_left_eef_target(action_step["left_wrist_eef"])
        # if "left_hand_joints" in action_step:
        #     self.sdk.set_left_hand_target(action_step["left_hand_joints"])
        return True
```

---

## 4. 카메라 매핑 및 설정

`REAL_R1_PRO_SHARPA`, `REAL_G1` 등 모델이 요구하는 비디오 키와 실제 물리 카메라 장치 ID를 매핑합니다:

| Embodiment Tag | 요구 Video Keys | 권장 카메라 위치 | 해상도 |
| :--- | :--- | :--- | :--- |
| **REAL_R1_PRO_SHARPA** | `ego_view_res320x240_freq20`<br>`left_wrist_view_res320x240_freq20`<br>`right_wrist_view_res320x240_freq20` | 머리/몸체 상단 (3인칭/시점)<br>왼쪽 손목 카메라<br>오른쪽 손목 카메라 | 320x240 |
| **REAL_G1** | `ego_view` | 머리 상단 (시점 카메라) | 224x224 |

* USB 웹캠 장치 번호 확인: `v4l2-ctl --list-devices` 또는 `/dev/video*`
* 실행 옵션: `--camera-devices 0 2 4` (순서대로 ego, left_wrist, right_wrist에 매핑)

---

## 5. 단계별 안전 테스트 프로토콜 (3-Stage Safety Protocol)

물리 로봇에 모터 토크를 인가하기 전에 반드시 아래 3단계를 순서대로 진행하여 안전을 검증하세요.

### **[1단계] 가상 환경 Mock 테스트 (모터/카메라 미연결 상태)**
하드웨어 없이 전체 비동기 파이프라인, 모델 로드, 추론 지연 시간을 검증합니다.
```bash
source .venv/bin/activate && source scripts/activate_spark.sh

python -m real_robot.run_real_robot \
    --embodiment-tag REAL_R1_PRO_SHARPA \
    --robot-backend mock \
    --camera-backend mock \
    --dry-run True
```

### **[2단계] 센서 연동 Dry-Run 테스트 (실제 카메라/센서 읽기, 모터 제어 차단)**
실제 카메라와 로봇 센서 피드백을 읽지만, **모터 구동 명령은 소프트웨어적으로 차단(`--dry-run True`)**하여 실시간 궤적이 안전한지 JSON으로 기록하고 확인합니다.
```bash
python -m real_robot.run_real_robot \
    --embodiment-tag REAL_R1_PRO_SHARPA \
    --robot-backend custom \
    --camera-backend opencv \
    --camera-devices 0 2 4 \
    --dry-run True \
    --save-action-json True \
    --save-action-json-path output_actions/dry_run_trajectory.json
```

### **[3단계] 실제 물리 로봇 클로즈드 루프 제어 (Live Execution)**
주변 안전 거리를 확보하고 비상 정지(E-Stop) 버튼을 준비한 뒤 모터를 인가합니다.
```bash
python -m real_robot.run_real_robot \
    --embodiment-tag REAL_R1_PRO_SHARPA \
    --robot-backend custom \
    --camera-backend opencv \
    --camera-devices 0 2 4 \
    --dry-run False \
    --control-freq 30.0 \
    --task-instruction "Pick up the red bottle and place it into the basket"
```

---

## 6. 하드웨어 보호 및 안전 기능 (`SafetyGuard`)

* **3D 작업공간 Bounding Box 제한:** 엔드이펙터가 바닥($z < -0.1m$)이나 몸체 뒤쪽으로 이동하지 못하도록 자동 클램핑.
* **급격한 충격 방지 (Velocity & Acceleration Limiting):** 타임스텝당 최대 변화량(예: $\Delta q \le 0.08\text{ rad/step}$)을 제한하여 관절이 튀는 현상 원천 차단.
* **SVD 특이점 보호:** 9D EEF Pose의 Rot6D 회전 벡터가 0으로 수렴하여 Unnormalization 특이점 에러가 발생하는 것을 방지.
* **비상 정지 (Emergency Stop):** `Ctrl+C` 입력 시 백그라운드 스레드를 즉시 종료하고 모터에 정지 명령 전송.
