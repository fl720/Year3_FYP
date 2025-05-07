"""
Rule-based fall detector driven by YOLO-Pose keypoints.

API
---
detector = RuleBasedFallDetector()
state    = detector.update(frame, ts)   # ts: float, seconds
"""
from collections import deque
from enum import Enum, auto
import numpy as np
import time
import torch
from ultralytics import YOLO

class PoseState(Enum):
    STAND    = auto()   # STANDING / NORMAL
    DESCENT  = auto()   # RAPID DESENDING / POTENTIAL FALL
    LYING    = auto()   # LYING
    FALL     = auto()   # FALLEN COMFIRMED

    def __str__(self):
        if self == PoseState.STAND:
            return "STAND"
        elif self == PoseState.DESCENT:
            return "DESCENT"
        elif self == PoseState.LYING:
            return "LYING"
        elif self == PoseState.FALL:
            return "FALL"
        else:
            return "UNKNOWN"


class RuleBasedFallDetector:
    """YOLO-Pose + Threshold + Fall Detection."""

    # ---------- KEY（COCO-17） ----------
    NOSE = 0
    L_HIP, R_HIP = 11, 12

    # ---------- INITIALSE ----------
    def __init__(
        self,
        low_fps: bool = True,  # low fps（5–10 fps）or high fps（≥25 fps）
        model_path: str = "./pose_demo/yolo11n-pose.pt",
        conf_thresh: float = 0.2,        # ignore if confidence level is below this threshold
        angle_upright_max: float = 30.0, # (°) angle between main body and the ground -> for standing
        angle_lying_min: float = 65.0,   # (°) angle between main body and the ground -> for lying
        ratio_upright_min: float = 1.2,  # h/w，stand
        ratio_lying_max: float = 0.5,    # h/w，just lying 
        v_speed_thresh: float = 0.25,    # speed threshold 
        lying_frames_confirm: int = 8,   # more than n frame -> FALL
        buffer_size: int = 30,           # save latest N frames
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = YOLO(model_path).to(device)
        self.model.fuse()
        self.device = device

        # ---------
        self.conf_th   = conf_thresh
        self.upr_ang   = np.radians(angle_upright_max)
        self.ly_ang    = np.radians(angle_lying_min)
        self.upr_ratio = ratio_upright_min
        self.ly_ratio  = ratio_lying_max
        self.vy_th     = v_speed_thresh
        self.ly_need   = lying_frames_confirm

        # ---------
        self.state      = PoseState.STAND
        self._buf       = deque(maxlen=buffer_size)  # Each: (mid_hip_y_norm, ts)
        self._lying_cnt = 0

        if low_fps:   # 5–10 fps
            self.upr_ang = np.radians(35)
            self.ly_ang  = np.radians(60)
            self.dy_th   = 0.20
            self.vy_th   = 0.15
            self.t_lying = 1.0
        else:         # ≥25 fps
            self.upr_ang = np.radians(30)
            self.ly_ang  = np.radians(65)
            self.dy_th   = 0.10
            self.vy_th   = 0.25
            self.t_lying = 0.6
        self.v_still = 0.05
        # extra save：0.5 s slide-window
        self._win_05s = deque()
        self._t_lying_start = None

    # ---------- external port ----------
    @torch.inference_mode()
    def update(self, frame: np.ndarray):
        ts = time.time()
        res, kpts, box = self._extract_pose(frame)
        if kpts is None:
            # 检测失败：按需求可选择返回上一次状态或 NONE
            return res, self.state

        torso_ang  = self._torso_angle(kpts)  # 弧度
        hw_ratio   = self._box_hw_ratio(box)  # h / w

        dy, vy = self._vertical_features(kpts, ts)
        self._state_machine(torso_ang, hw_ratio, dy, vy, ts)
        return res, self.state

    # ---------- 内部辅助 ----------
    def _extract_pose(self, frame):
        """返回关键点 ndarray (17,2) 和检测框 [x1,y1,x2,y2]; 若失败返回 (None, None, None)."""
        res = self.model.predict(frame, verbose=False)[0]
        kpts = res.keypoints.data
        if torch.isnan(kpts).any():
            kpts = torch.nan_to_num(kpts, nan=0.0)
            res.keypoints.data = kpts      

        if not res or res.keypoints.shape[0] == 0:
            return res, None, None

        # 仅取置信度最高的一个人
        idx = int(torch.argmax(res.boxes.conf))
        kpts = res.keypoints.xy[idx].cpu().numpy()       # (17,2)
        conf = res.keypoints.conf[idx].cpu().numpy()     # (17,)

        # 置信度过滤
        idx   = int(torch.argmax(res.boxes.conf))
        kpts  = res.keypoints.xy[idx].cpu().numpy()
        conf  = res.keypoints.conf[idx].cpu().numpy()
        kpts[conf < self.conf_th] = 0
        if np.isnan(kpts[[self.NOSE, self.L_HIP, self.R_HIP]]).any():
            return res, None, None

        box = res.boxes.xyxy[idx].cpu().numpy()          # (4,)
        return res, kpts, box

    def _torso_angle(self, kpts):
        """脖子(鼻尖近似)到髋部中心向量与竖直夹角 (rad)."""
        neck     = kpts[self.NOSE]
        mid_hip  = np.nanmean(kpts[[self.L_HIP, self.R_HIP]], axis=0)
        v        = neck - mid_hip
        ang      = np.arctan2(abs(v[0]), abs(v[1]))  # 与 y 轴夹角
        return ang

    def _box_hw_ratio(self, box):
        x1, y1, x2, y2 = box
        h, w = max(1.0, y2 - y1), max(1.0, x2 - x1)
        return h / w

    def _vertical_features(self, kpts, ts):
        mid_hip = np.nanmean(kpts[[self.L_HIP, self.R_HIP]], axis=0)
        y = mid_hip[1] / max(1.0, kpts[:, 1].max())
        self._buf.append((y, ts))
        # ---- 位移阈值 ----
        self._win_05s.append((y, ts))
        while ts - self._win_05s[0][1] > 0.5:
            self._win_05s.popleft()
        dy_05s = y - min(v for v, _ in self._win_05s)
        # ---- 中位速度 ----
        if len(self._buf) >= 3:
            vy = np.median([(self._buf[i+1][0]-self._buf[i][0]) /
                            (self._buf[i+1][1]-self._buf[i][1] + 1e-3)
                            for i in range(-3, -1)])
        else:
            vy = 0.0
        return dy_05s, vy
    
    def _state_machine(self, ang, ratio, dy, vy, ts):
        # -------- STAND --------
        if self.state == PoseState.STAND:
            if (dy > self.dy_th or vy > self.vy_th) and ang > self.upr_ang:
                self.state = PoseState.DESCENT

        # -------- DESCENT --------
        elif self.state == PoseState.DESCENT:
            if ang > self.ly_ang and ratio < self.ly_ratio:
                self.state = PoseState.LYING
                self._t_lying_start = ts            # 记录进入 LYING 的时刻
            elif ang < self.upr_ang:                # 误触
                self.state = PoseState.STAND

        # -------- LYING --------
        elif self.state == PoseState.LYING:
            still_lying = (
                ang > self.ly_ang and ratio < self.ly_ratio and vy < self.v_still
            )
            if still_lying:
                if ts - self._t_lying_start >= self.t_lying:
                    self.state = PoseState.FALL     # 满足“平躺 ≥ t_lying 秒”
            else:                                   # 起身或误判
                self.state = PoseState.STAND

        # -------- FALL --------
        elif self.state == PoseState.FALL:
            if ang < self.upr_ang and ratio > self.upr_ratio:
                self.state = PoseState.STAND

