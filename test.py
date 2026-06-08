import os
import cv2
import csv
import time
import argparse
import serial
import numpy as np
import mediapipe as mp


# ============================================================
# RH56 寄存器地址
# ============================================================
REG_ANGLE_SET = 0x05CE   # ANGLE_SET(m), 6short, 1486
REG_FORCE_SET = 0x05DA   # FORCE_SET(m), 6short, 1498
REG_ANGLE_ACT = 0x060A   # ANGLE_ACT(m), 6short, 1546
REG_FORCE_ACT = 0x062E   # FORCE_ACT(m), 6short, 1582
REG_STATUS    = 0x064C   # STATUS(m),    6byte,  1612
REG_CLEAR_ERR = 0x03EC   # CLEAR_ERROR,  1byte,  1004


# ============================================================
# RH56 6 自由度顺序
# ============================================================
# 说明书顺序：
# 0 小拇指
# 1 无名指
# 2 中指
# 3 食指
# 4 大拇指弯曲
# 5 大拇指旋转
#
# ANGLE_SET 是 0~1000 的寄存器值：
# 1000 通常表示张开方向
# 0    通常表示弯曲方向
#
# 注意：
# 这里不再直接用实际角度 19~176.7° 发送，
# 而是直接按说明书寄存器值 0~1000 控制，更适合力控抓取。
OPEN_VALUES = [1000, 1000, 1000, 1000, 1000, 1000]

# 普通全握闭合目标。
# 不建议一开始就全给 0，可以先给 80~150，力控会在接触时停止。
GRASP_CLOSE_VALUES = [80, 80, 80, 80, 80, 80]

# 捏取动作：
# 小拇指、无名指尽量张开；中指半开；食指和拇指参与捏取。
PINCH_CLOSE_VALUES = [1000, 1000, 800, 120, 120, 80]

# 三指抓取：
# 小拇指、无名指较开；中指、食指、拇指参与。
THREE_FINGER_CLOSE_VALUES = [1000, 900, 160, 120, 120, 80]


# ============================================================
# 基础工具
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def u16_to_le_bytes(v):
    v = int(v)
    if v < 0:
        # RH56 中 -1 表示该自由度不动作，对应 short 的 0xFFFF
        v = 0xFFFF
    v = clamp(v, 0, 0xFFFF)
    return [v & 0xFF, (v >> 8) & 0xFF]


def le_bytes_to_u16(lo, hi):
    return int(lo) | (int(hi) << 8)


def calc_checksum(cmd_without_checksum):
    return sum(cmd_without_checksum[2:]) & 0xFF


# ============================================================
# RS485 / RH56 控制器
# ============================================================
class RS485Driver:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=0.05):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout
        )

    def send(self, data: bytes):
        self.ser.write(data)

    def read(self, n=64):
        return self.ser.read(n)

    def reset_input_buffer(self):
        self.ser.reset_input_buffer()

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()


class RH56Controller:
    def __init__(
        self,
        port="/dev/ttyUSB0",
        baudrate=115200,
        hand_id=0x01,
        read_after_write=False,
    ):
        self.hand_id = int(hand_id)
        self.rs485 = RS485Driver(port=port, baudrate=baudrate)
        self.read_after_write = bool(read_after_write)

    def build_write_command(self, address, data_bytes):
        """
        写寄存器通用命令。
        帧格式：
        EB 90 ID LEN 12 ADDR_L ADDR_H DATA... CHECKSUM
        其中 LEN = len(DATA) + 3
        """
        addr_l = address & 0xFF
        addr_h = (address >> 8) & 0xFF
        data_len = len(data_bytes)

        cmd = [
            0xEB,
            0x90,
            self.hand_id,
            data_len + 3,
            0x12,
            addr_l,
            addr_h,
        ]

        cmd.extend(data_bytes)
        cmd.append(calc_checksum(cmd))
        return bytes(cmd)

    def build_read_command(self, address, read_len):
        """
        读寄存器通用命令。
        帧格式：
        EB 90 ID 04 11 ADDR_L ADDR_H READ_LEN CHECKSUM
        """
        addr_l = address & 0xFF
        addr_h = (address >> 8) & 0xFF

        cmd = [
            0xEB,
            0x90,
            self.hand_id,
            0x04,
            0x11,
            addr_l,
            addr_h,
            int(read_len),
        ]

        cmd.append(calc_checksum(cmd))
        return bytes(cmd)

    def write_bytes(self, address, data_bytes):
        cmd = self.build_write_command(address, data_bytes)

        try:
            self.rs485.reset_input_buffer()
        except Exception:
            pass

        self.rs485.send(cmd)

        ack = b""
        if self.read_after_write:
            time.sleep(0.01)
            ack = self.rs485.read(64)

        return cmd, ack

    def write_6short(self, address, values):
        """
        写连续 6 个 short 寄存器。
        values 长度必须为 6。
        """
        if len(values) != 6:
            raise ValueError("values 必须是长度为6的列表")

        data = []
        for v in values:
            data.extend(u16_to_le_bytes(v))

        return self.write_bytes(address, data)

    def write_1byte(self, address, value):
        value = int(clamp(value, 0, 255))
        return self.write_bytes(address, [value])

    def read_registers(self, address, read_len):
        """
        返回原始回复帧。
        """
        cmd = self.build_read_command(address, read_len)

        try:
            self.rs485.reset_input_buffer()
        except Exception:
            pass

        self.rs485.send(cmd)
        time.sleep(0.015)

        # 回复长度一般是：
        # 90 EB ID LEN 11 ADDR_L ADDR_H DATA... CHECKSUM
        # 总长度 = 8 + read_len
        expected = 8 + int(read_len)
        resp = self.rs485.read(expected + 16)

        return cmd, resp

    def parse_read_reply_data(self, resp, address, read_len):
        """
        从回复帧中取 DATA。
        正常回复：
        90 EB ID LEN 11 ADDR_L ADDR_H DATA... CHECKSUM
        """
        if resp is None or len(resp) < 8:
            return None

        data = list(resp)

        # 找帧头 90 EB
        start = -1
        for i in range(0, len(data) - 1):
            if data[i] == 0x90 and data[i + 1] == 0xEB:
                start = i
                break

        if start < 0:
            return None

        frame = data[start:]

        if len(frame) < 8:
            return None

        if frame[2] != self.hand_id:
            return None

        if frame[4] != 0x11:
            return None

        addr_l = address & 0xFF
        addr_h = (address >> 8) & 0xFF

        if frame[5] != addr_l or frame[6] != addr_h:
            return None

        if len(frame) < 7 + read_len:
            return None

        payload = frame[7:7 + read_len]
        return payload

    def read_6short(self, address):
        _, resp = self.read_registers(address, 12)
        payload = self.parse_read_reply_data(resp, address, 12)

        if payload is None or len(payload) < 12:
            return None

        values = []
        for i in range(0, 12, 2):
            values.append(le_bytes_to_u16(payload[i], payload[i + 1]))

        return values

    def read_6byte(self, address):
        _, resp = self.read_registers(address, 6)
        payload = self.parse_read_reply_data(resp, address, 6)

        if payload is None or len(payload) < 6:
            return None

        return [int(x) for x in payload[:6]]

    def set_force(self, force_values):
        """
        设置 FORCE_SET(m)，单位 g，范围 0~1000。
        """
        force_values = [int(clamp(v, 0, 1000)) for v in force_values]
        return self.write_6short(REG_FORCE_SET, force_values)

    def set_angle_values(self, angle_values):
        """
        设置 ANGLE_SET(m)，范围 -1 或 0~1000。
        """
        if len(angle_values) != 6:
            raise ValueError("angle_values 必须长度为6")

        safe_values = []
        for v in angle_values:
            if int(v) < 0:
                safe_values.append(-1)
            else:
                safe_values.append(int(clamp(v, 0, 1000)))

        return self.write_6short(REG_ANGLE_SET, safe_values)

    def open_hand(self):
        return self.set_angle_values(OPEN_VALUES)

    def force_grasp(self, force_values, close_values):
        """
        力控抓取：
        1. 先写 FORCE_SET
        2. 再给一个闭合方向 ANGLE_SET
        3. 接触后由灵巧手内部力控停止
        """
        self.set_force(force_values)
        time.sleep(0.04)
        return self.set_angle_values(close_values)

    def read_force_act(self):
        return self.read_6short(REG_FORCE_ACT)

    def read_angle_act(self):
        return self.read_6short(REG_ANGLE_ACT)

    def read_status(self):
        return self.read_6byte(REG_STATUS)

    def clear_error(self):
        return self.write_1byte(REG_CLEAR_ERR, 1)

    def close(self):
        self.rs485.close()


# ============================================================
# MediaPipe 手部计算函数
# ============================================================
def calc_angle_2d(a, b, c):
    """
    计算 ∠ABC，单位：度。
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    v1 = a - b
    v2 = c - b

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    cos_val = float(np.dot(v1, v2) / (n1 * n2))
    cos_val = np.clip(cos_val, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_val)))


def dist_2d(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def lm_to_px(hand_landmarks, w, h):
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float32)


def get_best_right_hand(results):
    """
    获取右手。
    如果左右手识别反了，可以运行时加 --mirror。
    """
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None, None

    best_idx = None
    best_score = -1.0

    for i, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label
        score = handedness.classification[0].score

        if label == "Right" and score > best_score:
            best_idx = i
            best_score = score

    if best_idx is None:
        return None, None

    return results.multi_hand_landmarks[best_idx], float(best_score)


# ============================================================
# 手势特征提取：从人手 21 点提取“动作意图”
# ============================================================
def finger_curl_from_landmarks(pts, ids):
    """
    ids = [mcp, pip, dip, tip]

    返回 curl:
        0 = 伸直
        1 = 弯曲
    """
    mcp, pip, dip, tip = ids

    pip_angle = calc_angle_2d(pts[mcp], pts[pip], pts[dip])
    dip_angle = calc_angle_2d(pts[pip], pts[dip], pts[tip])

    palm_width = dist_2d(pts[5], pts[17])
    if palm_width < 1e-6:
        palm_width = 1.0

    wrist = pts[0]
    tip_dist = dist_2d(pts[tip], wrist) / palm_width

    # PIP/DIP 越小越弯
    pip_curl = (165.0 - pip_angle) / (165.0 - 75.0)
    dip_curl = (165.0 - dip_angle) / (165.0 - 80.0)

    pip_curl = clamp(pip_curl, 0.0, 1.0)
    dip_curl = clamp(dip_curl, 0.0, 1.0)

    # 指尖越靠近手腕越弯。
    # 这个范围可以根据视频调：
    # 张开时 tip_dist 大，握拳时 tip_dist 小。
    tip_curl = (2.0 - tip_dist) / (2.0 - 0.9)
    tip_curl = clamp(tip_curl, 0.0, 1.0)

    curl = 0.50 * pip_curl + 0.20 * dip_curl + 0.30 * tip_curl
    curl = clamp(curl, 0.0, 1.0)

    debug = {
        "pip_angle": pip_angle,
        "dip_angle": dip_angle,
        "tip_dist": tip_dist,
        "pip_curl": pip_curl,
        "dip_curl": dip_curl,
        "tip_curl": tip_curl,
        "curl": curl,
    }

    return curl, debug


def thumb_features(pts):
    """
    计算拇指弯曲和对掌程度。
    """
    palm_width = dist_2d(pts[5], pts[17])
    if palm_width < 1e-6:
        palm_width = 1.0

    thumb_ip_angle = calc_angle_2d(pts[2], pts[3], pts[4])

    # 拇指自身弯曲程度
    thumb_bend = (165.0 - thumb_ip_angle) / (165.0 - 80.0)
    thumb_bend = clamp(thumb_bend, 0.0, 1.0)

    # 拇指尖到食指根部距离，越小越像对掌/捏取
    thumb_to_index_mcp = dist_2d(pts[4], pts[5]) / palm_width
    thumb_oppose = (1.4 - thumb_to_index_mcp) / (1.4 - 0.45)
    thumb_oppose = clamp(thumb_oppose, 0.0, 1.0)

    # 拇指尖到食指尖距离，越小越像捏取
    pinch_ratio = dist_2d(pts[4], pts[8]) / palm_width

    return {
        "thumb_ip_angle": thumb_ip_angle,
        "thumb_bend": thumb_bend,
        "thumb_to_index_mcp": thumb_to_index_mcp,
        "thumb_oppose": thumb_oppose,
        "pinch_ratio": pinch_ratio,
    }


def extract_hand_features(pts):
    """
    MediaPipe 21 点 -> 人手动作特征。
    """
    index_curl, index_dbg = finger_curl_from_landmarks(pts, [5, 6, 7, 8])
    middle_curl, middle_dbg = finger_curl_from_landmarks(pts, [9, 10, 11, 12])
    ring_curl, ring_dbg = finger_curl_from_landmarks(pts, [13, 14, 15, 16])
    pinky_curl, pinky_dbg = finger_curl_from_landmarks(pts, [17, 18, 19, 20])
    thumb_dbg = thumb_features(pts)

    avg_four_curl = float(np.mean([index_curl, middle_curl, ring_curl, pinky_curl]))
    avg_main_curl = float(np.mean([index_curl, middle_curl, ring_curl]))

    features = {
        "index_curl": index_curl,
        "middle_curl": middle_curl,
        "ring_curl": ring_curl,
        "pinky_curl": pinky_curl,
        "avg_four_curl": avg_four_curl,
        "avg_main_curl": avg_main_curl,
        "thumb_bend": thumb_dbg["thumb_bend"],
        "thumb_oppose": thumb_dbg["thumb_oppose"],
        "pinch_ratio": thumb_dbg["pinch_ratio"],
        "index_dbg": index_dbg,
        "middle_dbg": middle_dbg,
        "ring_dbg": ring_dbg,
        "pinky_dbg": pinky_dbg,
        "thumb_dbg": thumb_dbg,
    }

    return features


# ============================================================
# 手势意图状态机
# ============================================================
class IntentDebouncer:
    """
    防抖：
    raw_state 连续出现 stable_frames 帧后，才认为状态切换。
    """
    def __init__(self, stable_frames=5):
        self.stable_frames = int(max(1, stable_frames))
        self.last_raw = "unknown"
        self.raw_count = 0
        self.current_state = "unknown"

    def update(self, raw_state):
        raw_state = str(raw_state)

        if raw_state == self.last_raw:
            self.raw_count += 1
        else:
            self.last_raw = raw_state
            self.raw_count = 1

        changed = False

        if self.raw_count >= self.stable_frames and raw_state != self.current_state:
            old_state = self.current_state
            self.current_state = raw_state
            changed = True
            return self.current_state, changed, old_state

        return self.current_state, changed, self.current_state


class HandIntentClassifier:
    """
    将连续的人手特征转成离散动作意图：
        open
        grasp
        pinch
        three_finger
        hold
    """
    def __init__(
        self,
        open_curl_th=0.25,
        grasp_curl_th=0.62,
        pinch_ratio_th=0.45,
        three_finger_th=0.50,
    ):
        self.open_curl_th = float(open_curl_th)
        self.grasp_curl_th = float(grasp_curl_th)
        self.pinch_ratio_th = float(pinch_ratio_th)
        self.three_finger_th = float(three_finger_th)

    def classify(self, features):
        avg_four = features["avg_four_curl"]
        avg_main = features["avg_main_curl"]

        index_curl = features["index_curl"]
        middle_curl = features["middle_curl"]
        ring_curl = features["ring_curl"]
        pinky_curl = features["pinky_curl"]

        pinch_ratio = features["pinch_ratio"]
        thumb_oppose = features["thumb_oppose"]

        # 1. 张开：四指都比较直，拇指不明显对掌
        if avg_four < self.open_curl_th and pinch_ratio > 0.60:
            return "open"

        # 2. 捏取：拇指食指接近，食指有一定弯曲，小指/无名指不强制闭合
        if pinch_ratio < self.pinch_ratio_th and thumb_oppose > 0.35:
            return "pinch"

        # 3. 三指抓取：食指中指明显参与，拇指对掌，小指无名指相对更开
        if (
            thumb_oppose > 0.35
            and index_curl > self.three_finger_th
            and middle_curl > self.three_finger_th
            and pinky_curl < 0.55
        ):
            return "three_finger"

        # 4. 全握：四指平均弯曲较大
        if avg_four > self.grasp_curl_th or avg_main > self.grasp_curl_th:
            return "grasp"

        return "hold"


# ============================================================
# RH56 动作执行器：只在意图切换时发送命令
# ============================================================
class RH56ForceActionExecutor:
    def __init__(
        self,
        controller,
        no_serial=False,
        grasp_force=300,
        pinch_force=180,
        three_finger_force=250,
        thumb_force_scale=1.3,
        command_cooldown=0.8,
    ):
        self.controller = controller
        self.no_serial = bool(no_serial)

        self.grasp_force = int(grasp_force)
        self.pinch_force = int(pinch_force)
        self.three_finger_force = int(three_finger_force)
        self.thumb_force_scale = float(thumb_force_scale)

        self.command_cooldown = float(command_cooldown)
        self.last_command_time = 0.0
        self.last_executed_state = "unknown"

    def _force_list(self, base_force):
        base = int(clamp(base_force, 0, 1000))
        thumb = int(clamp(base * self.thumb_force_scale, 0, 1000))

        # 顺序：小、无、中、食、拇弯、拇旋
        return [base, base, base, base, thumb, base]

    def execute_if_needed(self, stable_state):
        now = time.time()

        if stable_state in ["unknown", "hold"]:
            return False, "no_action"

        if stable_state == self.last_executed_state:
            return False, "same_state"

        if now - self.last_command_time < self.command_cooldown:
            return False, "cooldown"

        command_desc = ""

        if stable_state == "open":
            command_desc = f"OPEN angle={OPEN_VALUES}"

            if not self.no_serial and self.controller is not None:
                self.controller.open_hand()

        elif stable_state == "grasp":
            force_values = self._force_list(self.grasp_force)
            close_values = GRASP_CLOSE_VALUES
            command_desc = f"FORCE_GRASP force={force_values} close={close_values}"

            if not self.no_serial and self.controller is not None:
                self.controller.force_grasp(force_values, close_values)

        elif stable_state == "pinch":
            force_values = self._force_list(self.pinch_force)
            close_values = PINCH_CLOSE_VALUES
            command_desc = f"FORCE_PINCH force={force_values} close={close_values}"

            if not self.no_serial and self.controller is not None:
                self.controller.force_grasp(force_values, close_values)

        elif stable_state == "three_finger":
            force_values = self._force_list(self.three_finger_force)
            close_values = THREE_FINGER_CLOSE_VALUES
            command_desc = f"FORCE_THREE_FINGER force={force_values} close={close_values}"

            if not self.no_serial and self.controller is not None:
                self.controller.force_grasp(force_values, close_values)

        else:
            return False, "unknown_state"

        self.last_executed_state = stable_state
        self.last_command_time = now

        return True, command_desc


# ============================================================
# 可视化
# ============================================================
def draw_hand_points(frame, pts):
    for i, p in enumerate(pts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        cv2.putText(
            frame,
            str(i),
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
        )

    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]

    for finger in fingers:
        for a, b in zip(finger[:-1], finger[1:]):
            pa = tuple(pts[a].astype(int))
            pb = tuple(pts[b].astype(int))
            cv2.line(frame, pa, pb, (0, 200, 255), 2)


def draw_status(
    frame,
    frame_idx,
    hand_score,
    raw_state,
    stable_state,
    features,
    command_sent,
    command_desc,
    send_enabled,
    force_act=None,
    status=None,
):
    lines = [
        f"frame: {frame_idx}",
        f"right hand score: {hand_score:.2f}",
        f"send RS485: {send_enabled}",
        f"raw_state: {raw_state}",
        f"stable_state: {stable_state}",
        f"curl I/M/R/P: "
        f"{features['index_curl']:.2f}, "
        f"{features['middle_curl']:.2f}, "
        f"{features['ring_curl']:.2f}, "
        f"{features['pinky_curl']:.2f}",
        f"avg_curl: {features['avg_four_curl']:.2f}",
        f"pinch_ratio: {features['pinch_ratio']:.2f}",
        f"thumb_bend/oppose: {features['thumb_bend']:.2f}, {features['thumb_oppose']:.2f}",
        f"command_sent: {command_sent}",
        f"command: {command_desc[:70]}",
    ]

    if force_act is not None:
        lines.append(f"FORCE_ACT: {force_act}")

    if status is not None:
        lines.append(f"STATUS: {status}")

    lines.append("keys: q/ESC=quit, s=toggle send, o=open, g=force grasp, p=pinch, e=clear error")

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            2,
        )
        y += 23


# ============================================================
# CSV 工具
# ============================================================
def write_csv(output_csv, rows):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 主流程：离线处理视频
# ============================================================
def run(args):
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"找不到视频文件: {args.video}")

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 1e-6:
        input_fps = args.output_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 80)
    print("[INFO] Offline RH56 force-control hand intent mapping")
    print(f"[INFO] video: {args.video}")
    print(f"[INFO] size: {width}x{height}")
    print(f"[INFO] fps: {input_fps}")
    print(f"[INFO] no_serial: {args.no_serial}")
    print("=" * 80)

    controller = None

    if args.no_serial:
        print("[INFO] no_serial 模式：只处理视频，不发送 RS485")
    else:
        controller = RH56Controller(
            port=args.port,
            baudrate=args.baudrate,
            hand_id=args.hand_id,
            read_after_write=args.read_after_write,
        )
        print(f"[INFO] RS485 opened: {args.port}, baudrate={args.baudrate}, hand_id={args.hand_id}")

    classifier = HandIntentClassifier(
        open_curl_th=args.open_curl_th,
        grasp_curl_th=args.grasp_curl_th,
        pinch_ratio_th=args.pinch_ratio_th,
        three_finger_th=args.three_finger_th,
    )

    debouncer = IntentDebouncer(
        stable_frames=args.stable_frames
    )

    executor = RH56ForceActionExecutor(
        controller=controller,
        no_serial=args.no_serial,
        grasp_force=args.grasp_force,
        pinch_force=args.pinch_force,
        three_finger_force=args.three_finger_force,
        thumb_force_scale=args.thumb_force_scale,
        command_cooldown=args.command_cooldown,
    )

    writer = None

    if args.save_vis:
        out_dir = os.path.dirname(args.output_video)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        writer = cv2.VideoWriter(
            args.output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            input_fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {args.output_video}")

    mp_hands = mp.solutions.hands

    rows = []
    frame_idx = 0
    send_enabled = not args.no_serial

    last_force_act = None
    last_status = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[INFO] 视频处理完成")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            hand_lms, hand_score = get_best_right_hand(results)

            row = {
                "frame_idx": frame_idx,
                "hand_detected": False,
                "hand_score": None,

                "raw_state": "no_hand",
                "stable_state": debouncer.current_state,
                "state_changed": False,

                "command_sent": False,
                "command_desc": "",

                "index_curl": None,
                "middle_curl": None,
                "ring_curl": None,
                "pinky_curl": None,
                "avg_four_curl": None,
                "avg_main_curl": None,
                "thumb_bend": None,
                "thumb_oppose": None,
                "pinch_ratio": None,

                "force_act": None,
                "status": None,
            }

            command_sent = False
            command_desc = "no_hand"
            raw_state = "no_hand"
            stable_state = debouncer.current_state
            state_changed = False

            if hand_lms is not None:
                pts = lm_to_px(hand_lms, w, h)
                features = extract_hand_features(pts)

                raw_state = classifier.classify(features)
                stable_state, state_changed, old_state = debouncer.update(raw_state)

                # 只有稳定状态切换时才发动作命令
                if send_enabled:
                    command_sent, command_desc = executor.execute_if_needed(stable_state)
                else:
                    command_sent, command_desc = False, "send_disabled"

                # 周期性读取实际受力和状态
                if (
                    not args.no_serial
                    and controller is not None
                    and args.read_feedback
                    and frame_idx % max(1, args.feedback_every_n) == 0
                ):
                    try:
                        last_force_act = controller.read_force_act()
                    except Exception as e:
                        print(f"[WARN] read FORCE_ACT failed: {e}")

                    try:
                        last_status = controller.read_status()
                    except Exception as e:
                        print(f"[WARN] read STATUS failed: {e}")

                draw_hand_points(frame, pts)
                draw_status(
                    frame=frame,
                    frame_idx=frame_idx,
                    hand_score=hand_score,
                    raw_state=raw_state,
                    stable_state=stable_state,
                    features=features,
                    command_sent=command_sent,
                    command_desc=command_desc,
                    send_enabled=send_enabled,
                    force_act=last_force_act,
                    status=last_status,
                )

                row.update({
                    "hand_detected": True,
                    "hand_score": hand_score,

                    "raw_state": raw_state,
                    "stable_state": stable_state,
                    "state_changed": state_changed,

                    "command_sent": command_sent,
                    "command_desc": command_desc,

                    "index_curl": features["index_curl"],
                    "middle_curl": features["middle_curl"],
                    "ring_curl": features["ring_curl"],
                    "pinky_curl": features["pinky_curl"],
                    "avg_four_curl": features["avg_four_curl"],
                    "avg_main_curl": features["avg_main_curl"],
                    "thumb_bend": features["thumb_bend"],
                    "thumb_oppose": features["thumb_oppose"],
                    "pinch_ratio": features["pinch_ratio"],

                    "force_act": "" if last_force_act is None else str(last_force_act),
                    "status": "" if last_status is None else str(last_status),
                })

            else:
                stable_state, state_changed, old_state = debouncer.update("no_hand")

                cv2.putText(
                    frame,
                    f"frame: {frame_idx} | No RIGHT hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"stable_state: {stable_state}",
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                row.update({
                    "raw_state": "no_hand",
                    "stable_state": stable_state,
                    "state_changed": state_changed,
                })

            rows.append(row)

            if writer is not None:
                writer.write(frame)

            if args.show:
                cv2.imshow("Offline RH56 Force Control", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    print("[INFO] 用户退出")
                    break

                elif key == ord("s"):
                    send_enabled = not send_enabled
                    print(f"[INFO] send_enabled = {send_enabled}")

                elif key == ord("o"):
                    print("[MANUAL CMD] OPEN")
                    if controller is not None and not args.no_serial:
                        controller.open_hand()

                elif key == ord("g"):
                    print("[MANUAL CMD] FORCE GRASP")
                    if controller is not None and not args.no_serial:
                        force_values = executor._force_list(args.grasp_force)
                        controller.force_grasp(force_values, GRASP_CLOSE_VALUES)

                elif key == ord("p"):
                    print("[MANUAL CMD] FORCE PINCH")
                    if controller is not None and not args.no_serial:
                        force_values = executor._force_list(args.pinch_force)
                        controller.force_grasp(force_values, PINCH_CLOSE_VALUES)

                elif key == ord("e"):
                    print("[MANUAL CMD] CLEAR ERROR")
                    if controller is not None and not args.no_serial:
                        controller.clear_error()

            print(
                f"[RUN] frame={frame_idx:06d} "
                f"hand={row['hand_detected']} "
                f"raw={row['raw_state']} "
                f"stable={row['stable_state']} "
                f"cmd={row['command_sent']} "
                f"{row['command_desc']}"
            )

            frame_idx += 1

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print("[INFO] 达到 max_frames，停止")
                break

    cap.release()

    if writer is not None:
        writer.release()

    if controller is not None:
        controller.close()

    cv2.destroyAllWindows()

    write_csv(args.output_csv, rows)

    print("=" * 80)
    print("[DONE] 离线视频 RH56 力控动作映射完成")
    print(f"[OK] CSV: {args.output_csv}")

    if args.save_vis:
        print(f"[OK] 可视化视频: {args.output_video}")

    print("=" * 80)


# ============================================================
# 参数
# ============================================================
def build_argparser():
    parser = argparse.ArgumentParser(
        description="Offline process recorded Orbbec color_video.mp4 with MediaPipe hand intent and RH56 force-control grasp."
    )

    parser.add_argument(
        "--video",
        type=str,
        default="/home/robot4/Programming/FoundationPose/video/5.22/color_video.mp4",
        help="已经采集好的彩色视频路径"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/robot4/Programming/FoundationPose/video/5.22/rh56_force_control_result.csv",
    )

    parser.add_argument(
        "--output_video",
        type=str,
        default="/home/robot4/Programming/FoundationPose/video/5.22/rh56_force_control_vis.mp4",
    )

    parser.add_argument("--save_vis", action="store_true", help="保存可视化视频")
    parser.add_argument("--show", action="store_true", help="显示处理窗口")
    parser.add_argument("--max_frames", type=int, default=0)

    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--hand_id", type=int, default=1)
    parser.add_argument("--no_serial", action="store_true")
    parser.add_argument("--read_after_write", action="store_true")

    parser.add_argument(
        "--read_feedback",
        action="store_true",
        help="周期性读取 FORCE_ACT 和 STATUS。串口设备不稳定时可以先不打开。",
    )
    parser.add_argument("--feedback_every_n", type=int, default=10)

    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--mirror", action="store_true")

    parser.add_argument("--min_detection_confidence", type=float, default=0.6)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.6)

    # 手势意图阈值
    parser.add_argument("--open_curl_th", type=float, default=0.25)
    parser.add_argument("--grasp_curl_th", type=float, default=0.62)
    parser.add_argument("--pinch_ratio_th", type=float, default=0.45)
    parser.add_argument("--three_finger_th", type=float, default=0.50)
    parser.add_argument("--stable_frames", type=int, default=5)

    # 力控参数，单位 g
    parser.add_argument("--grasp_force", type=int, default=300)
    parser.add_argument("--pinch_force", type=int, default=180)
    parser.add_argument("--three_finger_force", type=int, default=250)
    parser.add_argument("--thumb_force_scale", type=float, default=1.3)

    # 防止频繁重复发命令
    parser.add_argument("--command_cooldown", type=float, default=0.8)

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)