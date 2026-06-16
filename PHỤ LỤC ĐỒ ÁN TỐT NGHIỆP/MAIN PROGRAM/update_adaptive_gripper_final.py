#!/home/anhminh/yolo/bin/python3


from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import cv2
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
from ultralytics import YOLO

try:
    import pigpio
except Exception:
    pigpio = None

from tactile_sensor import TactileSensor
from update_force_model import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    GraspMLPModels,
    compute_tactile_features_from_sequence,
)

DEFAULT_YOLO_MODEL = "/home/anhminh/best_ncnn_model"
DEFAULT_SOURCE = "usb0"
DEFAULT_RESOLUTION = "640x480"
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_INFER_EVERY = 3
DEFAULT_TARGET_CLASS = ""
DEFAULT_TRIGGER_ZONE = "0.22,0.18,0.78,0.92"

DETECT_CONSECUTIVE_FRAMES = 3
STABLE_IOU_THRESH = 0.25
REARM_ONLY_WHEN_OBJECT_DISAPPEARS = True
MAX_DET = 12
IOU_THRESH = 0.45
MIN_BOX_AREA_RATIO = 0.012
CAMERA_FOURCC = "MJPG"
CAMERA_BUFFER = 1
CAMERA_FPS = 30

CALIBRATION_FILE = "servo_calibration.json"
RIGHT_PIN = 17
LEFT_PIN = 18
P_MIN = 500
P_MAX = 2500

BBOX_COLORS = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
    (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
    (98, 118, 150), (172, 176, 184),
]


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def parse_resolution(text: str) -> Tuple[int, int]:
    parts = str(text).lower().split("x")
    if len(parts) != 2:
        raise ValueError("resolution must be WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def parse_zone(text: str) -> Tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in str(text).split(",")]
    if len(vals) != 4:
        raise ValueError("trigger-zone must be x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    return (
        clamp(x1, 0.0, 1.0),
        clamp(y1, 0.0, 1.0),
        clamp(x2, 0.0, 1.0),
        clamp(y2, 0.0, 1.0),
    )


def zone_to_pixels(zone, width, height):
    x1, y1, x2, y2 = zone
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def open_camera(source: str, width: int, height: int, fourcc: str = CAMERA_FOURCC, fps: int = CAMERA_FPS, buffer_size: int = CAMERA_BUFFER):
    source = str(source)
    if source.startswith("usb"):
        cam_index = int(source[3:])
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    elif source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera/source: " + source)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))
    return cap


def draw_detection(frame, xyxy, class_name, conf, color):
    xmin, ymin, xmax, ymax = [int(v) for v in xyxy]
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    label = str(class_name) + ": " + str(int(float(conf) * 100)) + "%"
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_ymin = max(ymin, label_size[1] + 10)
    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def overlay_text(frame, lines, x=10, y=24, color=(0, 255, 255)):
    yy = int(y)
    for line in lines:
        cv2.putText(frame, str(line), (int(x), yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
        yy += 25


def extract_bbox_features(candidate: Mapping[str, Any], frame_shape) -> Dict[str, float]:
    h, w = frame_shape[:2]
    xmin, ymin, xmax, ymax = candidate["xyxy"]
    bw = max(1, int(xmax) - int(xmin))
    bh = max(1, int(ymax) - int(ymin))
    cx = (float(xmin) + float(xmax)) * 0.5
    cy = (float(ymin) + float(ymax)) * 0.5
    return {
        "bbox_w_norm": float(bw / float(w)),
        "bbox_h_norm": float(bh / float(h)),
        "bbox_area_norm": float((bw * bh) / float(w * h)),
        "aspect_ratio": float(bh / float(max(1, bw))),
        "cx_norm": float(cx / float(w)),
        "cy_norm": float(cy / float(h)),
        "conf": float(candidate.get("conf", 0.0)),
    }


def make_vision_features(candidate: Mapping[str, Any], frame_shape, object_instance_override: str = "") -> Dict[str, Any]:
    class_name = str(candidate.get("class_name", "unknown"))
    object_instance = object_instance_override.strip()
    if not object_instance:
        object_instance = class_name + "_unknown"
    features = {
        "object_class": class_name,
        "object_instance": object_instance,
    }
    features.update(extract_bbox_features(candidate, frame_shape))
    return features


class FrameGrabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = -1
        self.running = False
        self.thread = None
        self.slow_mode = False
        self.slow_interval_sec = 0.0

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.frame_id += 1
                slow_mode = self.slow_mode
                slow_interval_sec = self.slow_interval_sec
            if slow_mode and slow_interval_sec > 0.0:
                time.sleep(slow_interval_sec)

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return None, -1
            return self.frame.copy(), self.frame_id

    def set_slow_mode(self, enabled, interval_sec=0.10):
        with self.lock:
            self.slow_mode = bool(enabled)
            self.slow_interval_sec = float(max(0.0, interval_sec))

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class StableTrigger:
    def __init__(self, required_count: int = DETECT_CONSECUTIVE_FRAMES, iou_thresh: float = STABLE_IOU_THRESH):
        self.required_count = int(max(1, required_count))
        self.iou_thresh = float(clamp(iou_thresh, 0.0, 1.0))
        self.prev = None
        self.count = 0

    @staticmethod
    def bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / float(union)

    def reset(self):
        self.prev = None
        self.count = 0

    def update(self, candidate):
        if candidate is None:
            self.reset()
            return None
        if self.prev is None:
            self.count = 1
        else:
            same_class = candidate["class_name"] == self.prev["class_name"]
            iou = self.bbox_iou(candidate["xyxy"], self.prev["xyxy"])
            if same_class and iou >= self.iou_thresh:
                self.count += 1
            else:
                self.count = 1
        self.prev = candidate
        if self.count >= self.required_count:
            return candidate
        return None


def select_best_candidate(results, labels, global_thresh: float, target_class: str, frame_shape, min_box_area_ratio: float, zone_px):
    if not results:
        return None
    h, w = frame_shape[:2]
    frame_area = float(h * w)
    min_area = frame_area * float(min_box_area_ratio)
    zx1, zy1, zx2, zy2 = zone_px
    boxes = results[0].boxes
    candidates = []
    for det in boxes:
        conf = float(det.conf.item())
        class_idx = int(det.cls.item())
        class_name = str(labels[class_idx])
        if target_class and class_name != target_class:
            continue
        if conf < float(global_thresh):
            continue
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int).tolist()
        xmin, ymin, xmax, ymax = xyxy
        area = max(0, xmax - xmin) * max(0, ymax - ymin)
        if area < min_area:
            continue
        cx = int(round((xmin + xmax) / 2.0))
        cy = int(round((ymin + ymax) / 2.0))
        if not (zx1 <= cx <= zx2 and zy1 <= cy <= zy2):
            continue
        candidates.append({"conf": conf, "class_idx": class_idx, "class_name": class_name, "xyxy": xyxy, "area": area, "center": (cx, cy)})
    if not candidates:
        return None
    candidates.sort(key=lambda d: (d["conf"], d["area"]), reverse=True)
    return candidates[0]


class DualServoGripper:
    def __init__(self, calibration_file: str = CALIBRATION_FILE, hold_refresh_interval_sec: float = 0.12, hold_refresh_pulse_sec: float = 0.04, move_steps: int = 70):
        if pigpio is None:
            raise RuntimeError("pigpio Python module not found.")
        try:
            with open(calibration_file, "r", encoding="utf-8") as f:
                calib = json.load(f)
            self.right_open = int(calib[str(RIGHT_PIN)]["open"])
            self.right_close = int(calib[str(RIGHT_PIN)]["close"])
            self.left_open = int(calib[str(LEFT_PIN)]["open"])
            self.left_close = int(calib[str(LEFT_PIN)]["close"])
        except Exception:
            self.right_open, self.right_close = 30, 90
            self.left_open, self.left_close = 115, 55

        self.hold_refresh_interval_sec = float(max(0.05, hold_refresh_interval_sec))
        self.hold_refresh_pulse_sec = float(max(0.02, hold_refresh_pulse_sec))
        self.move_steps = int(max(5, move_steps))

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Cannot connect to pigpiod. Run: sudo pigpiod")

        self.pi.set_mode(RIGHT_PIN, pigpio.OUTPUT)
        self.pi.set_mode(LEFT_PIN, pigpio.OUTPUT)

        self.right_open_pulse = self.angle_to_pulse(self.right_open)
        self.right_close_pulse = self.angle_to_pulse(self.right_close)
        self.left_open_pulse = self.angle_to_pulse(self.left_open)
        self.left_close_pulse = self.angle_to_pulse(self.left_close)

        self.current_right_pulse = self.right_open_pulse
        self.current_left_pulse = self.left_open_pulse
        self.current_ratio = 0.0
        self.is_holding = False
        self.next_refresh_time = 0.0

    @staticmethod
    def angle_to_pulse(angle: int) -> int:
        angle = int(clamp(angle, 0, 180))
        ratio = angle / 180.0
        return int(round(P_MIN + ratio * (P_MAX - P_MIN)))

    def ratio_to_pulses(self, ratio: float) -> Tuple[int, int]:
        ratio = clamp(ratio, 0.0, 1.0)
        right = self.right_open_pulse + (self.right_close_pulse - self.right_open_pulse) * ratio
        left = self.left_open_pulse + (self.left_close_pulse - self.left_open_pulse) * ratio
        return int(round(right)), int(round(left))

    def set_pulse(self, pin: int, pulse: int):
        pulse = int(clamp(pulse, P_MIN, P_MAX))
        self.pi.set_servo_pulsewidth(pin, pulse)

    def release(self):
        self.pi.set_servo_pulsewidth(RIGHT_PIN, 0)
        self.pi.set_servo_pulsewidth(LEFT_PIN, 0)

    def move_both(self, target_right_pulse: int, target_left_pulse: int):
        steps = self.move_steps
        start_right = self.current_right_pulse
        start_left = self.current_left_pulse
        for i in range(steps + 1):
            progress = i / float(steps)
            ease = 0.5 - 0.5 * math.cos(math.pi * progress)
            right_pulse = start_right + (target_right_pulse - start_right) * ease
            left_pulse = start_left + (target_left_pulse - start_left) * ease
            self.set_pulse(RIGHT_PIN, int(round(right_pulse)))
            self.set_pulse(LEFT_PIN, int(round(left_pulse)))
            delay = 0.018 + (0.055 - 0.018) * (1.0 - ease)
            time.sleep(delay)
        self.set_pulse(RIGHT_PIN, int(target_right_pulse))
        self.set_pulse(LEFT_PIN, int(target_left_pulse))
        self.current_right_pulse = int(target_right_pulse)
        self.current_left_pulse = int(target_left_pulse)

    def open(self):
        self.is_holding = False
        right, left = self.ratio_to_pulses(0.0)
        self.move_both(right, left)
        self.current_ratio = 0.0
        self.release()

    def close_to_ratio(self, ratio: float):
        ratio = clamp(ratio, 0.0, 1.0)
        right, left = self.ratio_to_pulses(ratio)
        self.move_both(right, left)
        self.current_ratio = ratio
        self.is_holding = True
        self.next_refresh_time = time.monotonic() + self.hold_refresh_interval_sec
        self.release()

    def tick_hold(self):
        if not self.is_holding:
            return
        now = time.monotonic()
        if now < self.next_refresh_time:
            return
        self.set_pulse(RIGHT_PIN, self.current_right_pulse)
        self.set_pulse(LEFT_PIN, self.current_left_pulse)
        time.sleep(self.hold_refresh_pulse_sec)
        self.release()
        self.next_refresh_time = now + self.hold_refresh_interval_sec

    def cleanup(self):
        try:
            self.release()
            time.sleep(0.05)
        finally:
            try:
                self.pi.stop()
            except Exception:
                pass


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.auto_grasp = True
        self.pending_start = False
        self.pending_open = False
        self.pending_calibrate = False
        self.pending_rearm = False
        self.pending_quit = False
        self.target_class = ""
        self.object_instance = ""
        self.camera_frame_bgr = None
        self.runtime_state = "INIT"
        self.armed = True
        self.status_line = "Starting..."
        self.last_state = "none"
        self.last_action = "none"
        self.current_ratio = 0.0
        self.total_adjustments = 0
        self.no_contact_count = 0
        self.prob_success = 0.0
        self.prob_slip = 0.0
        self.prob_overforce = 0.0
        self.total_avg = 0.0
        self.total_max = 0.0
        self.contact_area_avg = 0.0
        self.tactile_snapshot = make_zero_tactile_snapshot()
        self.last_log = "No logs yet."

    def set_flag(self, name: str, value: bool = True):
        with self.lock:
            setattr(self, name, bool(value))

    def consume_flag(self, name: str) -> bool:
        with self.lock:
            value = bool(getattr(self, name))
            setattr(self, name, False)
            return value

    def update_runtime_values(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def get_snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "auto_grasp": self.auto_grasp,
                "target_class": self.target_class,
                "object_instance": self.object_instance,
                "camera_frame_bgr": self.camera_frame_bgr,
                "runtime_state": self.runtime_state,
                "armed": self.armed,
                "status_line": self.status_line,
                "last_state": self.last_state,
                "last_action": self.last_action,
                "current_ratio": self.current_ratio,
                "total_adjustments": self.total_adjustments,
                "no_contact_count": self.no_contact_count,
                "prob_success": self.prob_success,
                "prob_slip": self.prob_slip,
                "prob_overforce": self.prob_overforce,
                "total_avg": self.total_avg,
                "total_max": self.total_max,
                "contact_area_avg": self.contact_area_avg,
                "tactile_snapshot": json.loads(json.dumps(self.tactile_snapshot)),
                "last_log": self.last_log,
            }


def collect_tactile_window(sensor: TactileSensor, gripper: DualServoGripper, window_sec: float = 0.4, sample_interval_sec: float = 0.02):
    t0 = time.monotonic()
    sequence = []
    while (time.monotonic() - t0) < float(window_sec):
        gripper.tick_hold()
        forces = sensor.read_force()
        sequence.append(forces)
        if sample_interval_sec > 0:
            time.sleep(float(sample_interval_sec))
    features = compute_tactile_features_from_sequence(sequence)
    return sequence, features



class EventCSVLogger:
    

    FIELDS = [
        "timestamp",
        "elapsed_sec",
        "event_type",
        "runtime_state",
        "object_class",
        "object_instance",
        "current_ratio",
        "pred_state",
        "action",
        "p_success",
        "p_slip",
        "p_overforce",
        "total_avg",
        "total_max",
        "contact_area_avg",
        "no_contact_count",
        "total_adjustments",
        "message",
    ]

    def __init__(self, args):
        self.enabled = not bool(getattr(args, "disable_event_log", False))
        self.start_time = time.monotonic()
        self.file = None
        self.writer = None
        self.path = ""

        if not self.enabled:
            return

        log_file = str(getattr(args, "event_log_file", "") or "").strip()
        if log_file:
            path = Path(log_file).expanduser()
        else:
            log_dir = Path(str(getattr(args, "event_log_dir", "./event_logs"))).expanduser()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / ("adaptive_gripper_event_log_" + ts + ".csv")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)

        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.file.flush()

    def log(self, event_type="event", **kwargs):
        if not self.enabled or self.writer is None:
            return

        row = {field: "" for field in self.FIELDS}
        row["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
        row["elapsed_sec"] = "{:.3f}".format(time.monotonic() - self.start_time)
        row["event_type"] = str(event_type)

        for key, value in kwargs.items():
            if key not in row:
                continue
            if isinstance(value, float):
                row[key] = "{:.6f}".format(value)
            else:
                row[key] = "" if value is None else str(value)

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        try:
            if self.file is not None:
                self.file.flush()
                self.file.close()
        except Exception:
            pass


class AdaptiveRuntimeWorker(threading.Thread):
    def __init__(self, args, shared: SharedState):
        super().__init__(daemon=True)
        self.args = args
        self.shared = shared
        self.event_logger = EventCSVLogger(args)

    def log_event(self, event_type="event", **kwargs):
        try:
            self.event_logger.log(event_type=event_type, **kwargs)
        except Exception:
            pass

    def log(self, message: str, event_type: str = "runtime_log", **kwargs):
        print(message)
        self.shared.update_runtime_values(last_log=message, status_line=message)

        if "message" not in kwargs:
            kwargs["message"] = message
        self.log_event(event_type, **kwargs)

    def run(self):
        cap = None
        grabber = None
        sensor = None
        gripper = None
        try:
            res_w, res_h = parse_resolution(self.args.resolution)
            zone = parse_zone(self.args.trigger_zone)
            zone_px = zone_to_pixels(zone, res_w, res_h)

            if self.event_logger.enabled:
                self.log("[INFO] Event log CSV: " + self.event_logger.path, event_type="event_log_opened")
            else:
                self.log("[INFO] Event log CSV disabled.", event_type="event_log_disabled")

            self.log("[INFO] Loading YOLO...", event_type="startup")
            yolo = YOLO(self.args.yolo_model, task="detect")
            labels = yolo.names

            self.log("[INFO] Loading MLP model bundle...", event_type="model_loading")
            mlp = GraspMLPModels(model_path=self.args.model_path, metadata_path=self.args.metadata_path, verbose=True)
            policy = mlp.runtime_policy
            slip_step = float(policy.get("slip_increase_step", 0.02))
            overforce_step = float(policy.get("overforce_decrease_step", 0.01))
            metadata_max_adjustments = int(policy.get("max_adjustments", 8))
            if int(self.args.max_slip_adjustments) >= 0:
                max_slip_adjustments = int(self.args.max_slip_adjustments)
            else:
                max_slip_adjustments = metadata_max_adjustments
            max_overforce_decreases = int(max(1, self.args.max_overforce_decreases))

            self.log("[INFO] Initialising tactile sensor...", event_type="sensor_init")
            sensor = TactileSensor()
            if not self.args.skip_calibration:
                self.log("[INFO] Calibrating tactile baseline...", event_type="calibration_start")
                sensor.calibrate()
                self.log("[INFO] Tactile calibration complete.", event_type="calibration_done")

            self.log("[INFO] Initialising gripper...", event_type="gripper_init")
            gripper = DualServoGripper()
            gripper.open()

            self.log("[INFO] Opening camera...", event_type="camera_open")
            cap = open_camera(self.args.source, res_w, res_h, self.args.camera_fourcc, self.args.camera_fps, self.args.camera_buffer)
            grabber = FrameGrabber(cap).start()
            stable_filter = StableTrigger(required_count=self.args.stable_count, iou_thresh=self.args.stable_iou)

            runtime_state = "OPEN"
            armed = True
            last_seen_frame_id = -1
            last_infer_frame_id = -1
            new_frame_counter = 0
            best_candidate = None
            stable_candidate = None
            active_vision_features = None
            current_ratio = 0.0
            consecutive_slip_adjustments = 0
            persistent_slip_after_max = 0
            consecutive_overforce_decreases = 0
            persistent_overforce_after_max = 0
            total_adjustments = 0
            no_contact_count = 0

            self.shared.update_runtime_values(runtime_state=runtime_state, armed=armed)
            self.log("[INFO] UI runtime ready.", event_type="runtime_ready")

            while self.shared.running:
                if self.shared.consume_flag("pending_quit"):
                    self.shared.running = False
                    break

                if self.shared.consume_flag("pending_open"):
                    self.log("[INFO] Manual open/release.", event_type="manual_open")
                    gripper.open()
                    runtime_state = "OPEN"
                    grabber.set_slow_mode(False)
                    armed = True
                    stable_filter.reset()
                    best_candidate = None
                    stable_candidate = None
                    active_vision_features = None
                    current_ratio = 0.0
                    consecutive_slip_adjustments = 0
                    persistent_slip_after_max = 0
                    consecutive_overforce_decreases = 0
                    persistent_overforce_after_max = 0
                    total_adjustments = 0
                    no_contact_count = 0
                    self.shared.update_runtime_values(runtime_state=runtime_state, armed=armed, current_ratio=current_ratio, total_adjustments=total_adjustments, no_contact_count=no_contact_count, last_state="none", last_action="manual_open")
                    clear_tactile_runtime_values(self.shared)

                if self.shared.consume_flag("pending_rearm"):
                    self.log("[INFO] Rearmed.", event_type="rearm")
                    armed = True
                    stable_filter.reset()
                    best_candidate = None
                    stable_candidate = None
                    self.shared.update_runtime_values(armed=armed)

                if self.shared.consume_flag("pending_calibrate"):
                    self.log("[INFO] Recalibrating tactile baseline...", event_type="calibration_start")
                    gripper.open()
                    runtime_state = "OPEN"
                    grabber.set_slow_mode(False)
                    armed = True
                    stable_filter.reset()
                    best_candidate = None
                    stable_candidate = None
                    active_vision_features = None
                    current_ratio = 0.0
                    no_contact_count = 0
                    sensor.calibrate()
                    self.log("[INFO] Calibration complete.", event_type="calibration_done")
                    self.shared.update_runtime_values(runtime_state=runtime_state, armed=armed, current_ratio=current_ratio, no_contact_count=no_contact_count)
                    clear_tactile_runtime_values(self.shared)

                frame, frame_id = grabber.read_latest()
                if frame is None:
                    time.sleep(0.005)
                    continue

                if frame_id != last_seen_frame_id:
                    last_seen_frame_id = frame_id
                    new_frame_counter += 1

                if frame.shape[1] != res_w or frame.shape[0] != res_h:
                    frame = cv2.resize(frame, (res_w, res_h))

                display = frame.copy()
                if runtime_state == "HOLDING":
                    gripper.tick_hold()

                target_class = self.shared.target_class.strip()
                object_instance = self.shared.object_instance.strip()
                auto_grasp = bool(self.shared.auto_grasp)

                should_infer = (
                    runtime_state == "OPEN"
                    and frame_id != last_infer_frame_id
                    and (new_frame_counter % max(1, self.args.infer_every) == 0)
                )

                if should_infer:
                    results = yolo.predict(source=frame, imgsz=self.args.imgsz, conf=self.args.conf, iou=self.args.iou, max_det=self.args.max_det, verbose=False)
                    last_infer_frame_id = frame_id
                    best_candidate = select_best_candidate(results, labels, self.args.conf, target_class, frame.shape, self.args.min_box_area_ratio, zone_px)
                    stable_candidate = stable_filter.update(best_candidate)

                if runtime_state == "OPEN" and best_candidate is not None:
                    color = BBOX_COLORS[best_candidate["class_idx"] % len(BBOX_COLORS)]
                    draw_detection(display, best_candidate["xyxy"], best_candidate["class_name"], best_candidate["conf"], color)

                cv2.rectangle(display, (zone_px[0], zone_px[1]), (zone_px[2], zone_px[3]), (0, 255, 0), 2)
                stable_text = "READY" if stable_candidate is not None else f"{stable_filter.count}/{self.args.stable_count}"

                if runtime_state == "OPEN":
                   
                    overlay_text(display, [
                        f"Stable: {stable_text}",
                        f"Target class: {target_class if target_class else 'all'}",
                        f"Object instance: {object_instance if object_instance else 'class_unknown'}",
                    ], x=10, y=24, color=(0, 255, 255))
                else:
                    
                    pass

                self.shared.update_runtime_values(camera_frame_bgr=display, runtime_state=runtime_state, armed=armed)

                if runtime_state == "OPEN" and best_candidate is None and REARM_ONLY_WHEN_OBJECT_DISAPPEARS:
                    armed = True

                start_grasp = False
                if runtime_state == "OPEN" and armed:
                    if auto_grasp and stable_candidate is not None:
                        start_grasp = True
                    elif self.shared.consume_flag("pending_start"):
                        start_grasp = True

                if start_grasp:
                    if stable_candidate is None:
                        self.log("[WARN] No stable detection yet.")
                    else:
                        active_vision_features = make_vision_features(stable_candidate, frame.shape, object_instance_override=object_instance)
                        initial = mlp.predict_initial_grip_ratio(active_vision_features)
                        current_ratio = float(initial["initial_grip_ratio"])
                        grasp_msg = "[GRASP] class={} instance={} initial_ratio={:.4f}".format(
                            active_vision_features["object_class"],
                            active_vision_features["object_instance"],
                            current_ratio,
                        )
                        self.log(
                            grasp_msg,
                            event_type="grasp_start",
                            runtime_state=runtime_state,
                            object_class=active_vision_features["object_class"],
                            object_instance=active_vision_features["object_instance"],
                            current_ratio=current_ratio,
                        )
                        gripper.close_to_ratio(current_ratio)
                        runtime_state = "HOLDING"
                        if self.args.holding_camera_sleep > 0.0:
                            grabber.set_slow_mode(True, self.args.holding_camera_sleep)
                        armed = False
                        stable_filter.reset()
                        best_candidate = None
                        stable_candidate = None
                        consecutive_slip_adjustments = 0
                        persistent_slip_after_max = 0
                        consecutive_overforce_decreases = 0
                        persistent_overforce_after_max = 0
                        total_adjustments = 0
                        no_contact_count = 0
                        self.shared.update_runtime_values(runtime_state=runtime_state, armed=armed, current_ratio=current_ratio, total_adjustments=total_adjustments, no_contact_count=no_contact_count, last_state="starting_hold", last_action="close_initial")
                        continue

                if runtime_state == "HOLDING" and active_vision_features is not None:
                    sequence, tactile_features = collect_tactile_window(sensor, gripper, self.args.window_sec, self.args.sample_interval)
                    tactile_snapshot = sequence[-1] if sequence else self.shared.tactile_snapshot
                    model_state = mlp.predict_tactile_state(vision_features=active_vision_features, grip_ratio=current_ratio, tactile_features=tactile_features)
                    probs = model_state["probabilities"]
                    action = str(model_state["action"])
                    pred_state = str(model_state["state"])
                    total_max_now = float(tactile_features.get("total_max", 0.0))
                    contact_area_now = float(tactile_features.get("contact_area_avg", 0.0))
                    no_contact = total_max_now <= float(self.args.no_contact_total_max) and contact_area_now <= float(self.args.no_contact_area)
                    if no_contact:
                        no_contact_count += 1
                    else:
                        no_contact_count = 0

                    self.shared.update_runtime_values(
                        tactile_snapshot=tactile_snapshot,
                        last_state=pred_state,
                        last_action=action,
                        prob_success=float(probs.get("success", 0.0)),
                        prob_slip=float(probs.get("slip", 0.0)),
                        prob_overforce=float(probs.get("overforce", 0.0)),
                        total_avg=float(tactile_features.get("total_avg", 0.0)),
                        total_max=total_max_now,
                        contact_area_avg=contact_area_now,
                        no_contact_count=no_contact_count,
                        current_ratio=current_ratio,
                        total_adjustments=total_adjustments,
                    )

                    state_msg = "[STATE] ratio={:.3f} pred={} action={} p_success={:.2f} p_slip={:.2f} p_overforce={:.2f} total_avg={:.1f} total_max={:.1f} area={:.3f}".format(
                        current_ratio,
                        pred_state,
                        action,
                        probs.get("success", 0.0),
                        probs.get("slip", 0.0),
                        probs.get("overforce", 0.0),
                        float(tactile_features.get("total_avg", 0.0)),
                        total_max_now,
                        contact_area_now,
                    )
                    self.log(
                        state_msg,
                        event_type="state_prediction",
                        runtime_state=runtime_state,
                        object_class=active_vision_features.get("object_class", ""),
                        object_instance=active_vision_features.get("object_instance", ""),
                        current_ratio=current_ratio,
                        pred_state=pred_state,
                        action=action,
                        p_success=float(probs.get("success", 0.0)),
                        p_slip=float(probs.get("slip", 0.0)),
                        p_overforce=float(probs.get("overforce", 0.0)),
                        total_avg=float(tactile_features.get("total_avg", 0.0)),
                        total_max=total_max_now,
                        contact_area_avg=contact_area_now,
                        no_contact_count=no_contact_count,
                        total_adjustments=total_adjustments,
                    )

                    if no_contact_count >= int(max(1, self.args.no_contact_windows)):
                        self.log("[FAIL] No contact/object lost. Opening and resetting.", event_type="object_lost", runtime_state=runtime_state, current_ratio=current_ratio, no_contact_count=no_contact_count, total_adjustments=total_adjustments)
                        gripper.open()
                        runtime_state = "OPEN"
                        grabber.set_slow_mode(False)
                        armed = True
                        stable_filter.reset()
                        best_candidate = None
                        stable_candidate = None
                        active_vision_features = None
                        current_ratio = 0.0
                        consecutive_slip_adjustments = 0
                        persistent_slip_after_max = 0
                        consecutive_overforce_decreases = 0
                        persistent_overforce_after_max = 0
                        total_adjustments = 0
                        no_contact_count = 0
                        self.shared.update_runtime_values(runtime_state=runtime_state, armed=armed, current_ratio=current_ratio, total_adjustments=total_adjustments, no_contact_count=no_contact_count, last_state="object_lost", last_action="auto_open")
                        clear_tactile_runtime_values(self.shared)
                        continue

                    if action == "increase":
                        consecutive_overforce_decreases = 0
                        persistent_overforce_after_max = 0
                        if consecutive_slip_adjustments < max_slip_adjustments:
                            next_ratio = clamp(current_ratio + abs(slip_step), 0.0, 1.0)
                            if abs(next_ratio - current_ratio) < 1e-6:
                                self.log("[GRASP] Upper ratio limit reached during slip recovery.", event_type="slip_ratio_limit", runtime_state=runtime_state, current_ratio=current_ratio, total_adjustments=total_adjustments)
                                persistent_slip_after_max += 1
                            else:
                                self.log(
                                    "[GRASP] Slip recovery ratio {:.3f} -> {:.3f}".format(current_ratio, next_ratio),
                                    event_type="slip_adjustment",
                                    runtime_state=runtime_state,
                                    object_class=active_vision_features.get("object_class", ""),
                                    object_instance=active_vision_features.get("object_instance", ""),
                                    current_ratio=next_ratio,
                                    pred_state=pred_state,
                                    action=action,
                                    p_success=float(probs.get("success", 0.0)),
                                    p_slip=float(probs.get("slip", 0.0)),
                                    p_overforce=float(probs.get("overforce", 0.0)),
                                    total_avg=float(tactile_features.get("total_avg", 0.0)),
                                    total_max=total_max_now,
                                    contact_area_avg=contact_area_now,
                                    no_contact_count=no_contact_count,
                                    total_adjustments=total_adjustments + 1,
                                )
                                gripper.close_to_ratio(next_ratio)
                                current_ratio = next_ratio
                                consecutive_slip_adjustments += 1
                                total_adjustments += 1
                                persistent_slip_after_max = 0
                        else:
                            persistent_slip_after_max += 1
                            self.log(
                                "[GRASP] Max slip adjustments reached. persistent_slip_after_max={}/{}".format(persistent_slip_after_max, int(self.args.persistent_slip_windows_after_max)),
                                event_type="max_slip_adjustments",
                                runtime_state=runtime_state,
                                current_ratio=current_ratio,
                                pred_state=pred_state,
                                action=action,
                                p_success=float(probs.get("success", 0.0)),
                                p_slip=float(probs.get("slip", 0.0)),
                                p_overforce=float(probs.get("overforce", 0.0)),
                                total_avg=float(tactile_features.get("total_avg", 0.0)),
                                total_max=total_max_now,
                                contact_area_avg=contact_area_now,
                                no_contact_count=no_contact_count,
                                total_adjustments=total_adjustments,
                            )

                        if persistent_slip_after_max >= int(max(1, self.args.persistent_slip_windows_after_max)):
                            self.log("[FAIL] Persistent slip after max adjustments. Opening and resetting.", event_type="grasp_unstable", runtime_state=runtime_state, current_ratio=current_ratio, total_adjustments=total_adjustments)
                            gripper.open()
                            runtime_state = "OPEN"
                            grabber.set_slow_mode(False)
                            armed = True
                            stable_filter.reset()
                            best_candidate = None
                            stable_candidate = None
                            active_vision_features = None
                            current_ratio = 0.0
                            consecutive_slip_adjustments = 0
                            persistent_slip_after_max = 0
                            consecutive_overforce_decreases = 0
                            persistent_overforce_after_max = 0
                            total_adjustments = 0
                            no_contact_count = 0
                            self.shared.update_runtime_values(runtime_state=runtime_state, armed=armed, current_ratio=current_ratio, total_adjustments=total_adjustments, no_contact_count=no_contact_count, last_state="grasp_unstable", last_action="auto_open")
                            clear_tactile_runtime_values(self.shared)
                            continue

                    elif action == "decrease":
                        consecutive_slip_adjustments = 0
                        persistent_slip_after_max = 0
                        if consecutive_overforce_decreases < max_overforce_decreases:
                            next_ratio = clamp(current_ratio - abs(overforce_step), 0.0, 1.0)
                            if abs(next_ratio - current_ratio) < 1e-6:
                                self.log("[GRASP] Lower ratio limit reached during overforce recovery.", event_type="overforce_ratio_limit", runtime_state=runtime_state, current_ratio=current_ratio, total_adjustments=total_adjustments)
                                persistent_overforce_after_max += 1
                            else:
                                self.log(
                                    "[GRASP] Overforce safety ratio {:.3f} -> {:.3f}".format(current_ratio, next_ratio),
                                    event_type="overforce_adjustment",
                                    runtime_state=runtime_state,
                                    object_class=active_vision_features.get("object_class", ""),
                                    object_instance=active_vision_features.get("object_instance", ""),
                                    current_ratio=next_ratio,
                                    pred_state=pred_state,
                                    action=action,
                                    p_success=float(probs.get("success", 0.0)),
                                    p_slip=float(probs.get("slip", 0.0)),
                                    p_overforce=float(probs.get("overforce", 0.0)),
                                    total_avg=float(tactile_features.get("total_avg", 0.0)),
                                    total_max=total_max_now,
                                    contact_area_avg=contact_area_now,
                                    no_contact_count=no_contact_count,
                                    total_adjustments=total_adjustments + 1,
                                )
                                gripper.close_to_ratio(next_ratio)
                                current_ratio = next_ratio
                                consecutive_overforce_decreases += 1
                                total_adjustments += 1
                                persistent_overforce_after_max = 0
                        else:
                            persistent_overforce_after_max += 1
                            self.log(
                                "[SAFETY] Max overforce decreases reached. persistent_overforce_after_max={}/{}".format(persistent_overforce_after_max, int(self.args.persistent_overforce_windows_after_max)),
                                event_type="max_overforce_decreases",
                                runtime_state=runtime_state,
                                current_ratio=current_ratio,
                                pred_state=pred_state,
                                action=action,
                                p_success=float(probs.get("success", 0.0)),
                                p_slip=float(probs.get("slip", 0.0)),
                                p_overforce=float(probs.get("overforce", 0.0)),
                                total_avg=float(tactile_features.get("total_avg", 0.0)),
                                total_max=total_max_now,
                                contact_area_avg=contact_area_now,
                                no_contact_count=no_contact_count,
                                total_adjustments=total_adjustments,
                            )

                        if persistent_overforce_after_max >= int(max(1, self.args.persistent_overforce_windows_after_max)):
                            self.log("[SAFETY] Persistent overforce. Opening and resetting.", event_type="overforce_safety_open", runtime_state=runtime_state, current_ratio=current_ratio, total_adjustments=total_adjustments)
                            gripper.open()
                            runtime_state = "OPEN"
                            grabber.set_slow_mode(False)
                            armed = True
                            stable_filter.reset()
                            best_candidate = None
                            stable_candidate = None
                            active_vision_features = None
                            current_ratio = 0.0
                            consecutive_slip_adjustments = 0
                            persistent_slip_after_max = 0
                            consecutive_overforce_decreases = 0
                            persistent_overforce_after_max = 0
                            total_adjustments = 0
                            no_contact_count = 0
                            self.shared.update_runtime_values(runtime_state=runtime_state, armed=armed, current_ratio=current_ratio, total_adjustments=total_adjustments, no_contact_count=no_contact_count, last_state="overforce_safety_open", last_action="auto_open")
                            clear_tactile_runtime_values(self.shared)
                            continue

                    else:
                        consecutive_slip_adjustments = 0
                        persistent_slip_after_max = 0
                        consecutive_overforce_decreases = 0
                        persistent_overforce_after_max = 0

                    self.shared.update_runtime_values(current_ratio=current_ratio, total_adjustments=total_adjustments)

                time.sleep(0.005)

        except Exception as exc:
            self.log("[ERROR] " + str(exc), event_type="error", message=str(exc))
            self.shared.running = False

        finally:
            self.log("[INFO] Cleaning up runtime...", event_type="cleanup")
            try:
                if gripper is not None:
                    gripper.open()
            except Exception:
                pass
            try:
                if gripper is not None:
                    gripper.cleanup()
            except Exception:
                pass
            try:
                if sensor is not None:
                    sensor.cleanup()
            except Exception:
                pass
            try:
                if grabber is not None:
                    grabber.stop()
            except Exception:
                pass
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                self.log_event("event_log_closed", message="Event log closed")
                self.event_logger.close()
            except Exception:
                pass
            self.shared.running = False




HAND_BG_COLOR = "#eeeeee"
HAND_LOW_COLOR = "#4b45a5"
HAND_MANUAL_MAX = 22000.0
HAND_AUTO_SCALE_FLOOR = 1200.0
HAND_AUTO_SCALE_MARGIN = 1.20
HAND_AUTO_SCALE_DECAY = 0.97
HAND_COLOR_GAMMA = 1.0


HAND_DISPLAY_SCALE = 1.30

HAND_SENSORS = [
    {
        "key": "L_OUT",
        "label": "outer 3x2",
        "rows": 3,
        "cols": 2,
        "rotate": 90,
        "x": 640,
        "y": 145,
        "w": 150,
        "h": 100,
    },
    {
        "key": "L_IN",
        "label": "inner 3x3",
        "rows": 3,
        "cols": 3,
        "rotate": 90,
        "x": 455,
        "y": 105,
        "w": 180,
        "h": 180,
    },
    {
        "key": "R_IN",
        "label": "inner 3x3",
        "rows": 3,
        "cols": 3,
        "rotate": 270,
        "x": 190,
        "y": 105,
        "w": 180,
        "h": 180,
    },
    {
        "key": "R_OUT",
        "label": "outer 3x2",
        "rows": 3,
        "cols": 2,
        "rotate": 270,
        "x": 35,
        "y": 145,
        "w": 150,
        "h": 100,
    },
]


def make_zero_tactile_snapshot():
    
    return {
        "L_OUT": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "L_IN": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "R_IN": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "R_OUT": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    }


def clear_tactile_runtime_values(shared_state):
    
    shared_state.update_runtime_values(
        tactile_snapshot=make_zero_tactile_snapshot(),
        total_avg=0.0,
        total_max=0.0,
        contact_area_avg=0.0,
        prob_success=0.0,
        prob_slip=0.0,
        prob_overforce=0.0,
        no_contact_count=0,
    )


class HandDisplayCanvas:
   

    def __init__(self, parent):
   
        self.frame = tk.Frame(parent, bg=HAND_BG_COLOR)
        self.auto_scale = tk.BooleanVar(value=False)
        self.gui_scale_max = float(HAND_MANUAL_MAX)

        self.cell_items = {}
        self.text_items = {}
        self.info_items = {}
        self.display_values = {}
        self.smooth_alpha = 0.35

        for sensor in HAND_SENSORS:
            key = sensor["key"]
            rows = sensor["rows"]
            cols = sensor["cols"]
            self.display_values[key] = [[0.0 for _ in range(cols)] for _ in range(rows)]

        top = tk.Frame(self.frame, bg=HAND_BG_COLOR)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_dummy = tk.Button(top, text="Stop", width=10, state=tk.DISABLED)
        self.btn_dummy.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_cal_dummy = tk.Button(top, text="Calibrate zero", width=14, state=tk.DISABLED)
        self.btn_cal_dummy.pack(side=tk.LEFT, padx=5, pady=5)

        self.chk_auto = tk.Checkbutton(top, text="Auto scale", variable=self.auto_scale, bg=HAND_BG_COLOR)
        self.chk_auto.pack(side=tk.LEFT, padx=5, pady=5)

        self.status = tk.Label(top, text="Runtime tactile display", bg=HAND_BG_COLOR, anchor="w")
        self.status.pack(side=tk.LEFT, padx=10, pady=5)

        canvas_w = int(860 * HAND_DISPLAY_SCALE)
        canvas_h = int(350 * HAND_DISPLAY_SCALE)

        self.canvas = tk.Canvas(
            self.frame,
            width=canvas_w,
            height=canvas_h,
            bg=HAND_BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.TOP, anchor="center", pady=(4, 0))

        self.draw_layout()

        
        if HAND_DISPLAY_SCALE != 1.0:
            self.canvas.scale("all", 0, 0, HAND_DISPLAY_SCALE, HAND_DISPLAY_SCALE)

    def draw_layout(self):
        self.canvas.create_text(205, 45, text="RIGHT FINGER", font=("Arial", 12, "bold"), fill="black")
        self.canvas.create_text(610, 45, text="LEFT FINGER", font=("Arial", 12, "bold"), fill="black")

        for sensor in HAND_SENSORS:
            self.draw_sensor(sensor)

        self.draw_colorbar()

    def get_display_grid_size(self, rows, cols, rotate):
        rotate = rotate % 360
        if rotate == 90 or rotate == 270:
            return cols, rows
        return rows, cols

    def map_cell_to_display(self, r, c, rows, cols, rotate):
        rotate = rotate % 360
        if rotate == 90:
            return c, rows - 1 - r
        if rotate == 180:
            return rows - 1 - r, cols - 1 - c
        if rotate == 270:
            return cols - 1 - c, r
        return r, c

    def draw_sensor(self, sensor):
        key = sensor["key"]
        rows = sensor["rows"]
        cols = sensor["cols"]
        x = sensor["x"]
        y = sensor["y"]
        w = sensor["w"]
        h = sensor["h"]

        self.info_items[key] = self.canvas.create_text(
            x,
            y - 34,
            text="sum: 0\npeak: 0",
            anchor="w",
            font=("Arial", 8),
            fill="black",
        )

        self.canvas.create_rectangle(x - 2, y - 2, x + w + 2, y + h + 2, fill="white", outline="white")

        self.cell_items[key] = []
        self.text_items[key] = []

        rotate = sensor.get("rotate", 0)
        disp_rows, disp_cols = self.get_display_grid_size(rows, cols, rotate)
        cell_w = float(w) / float(disp_cols)
        cell_h = float(h) / float(disp_rows)

        for r in range(rows):
            rect_row = []
            text_row = []
            for c in range(cols):
                dr, dc = self.map_cell_to_display(r, c, rows, cols, rotate)
                x1 = x + dc * cell_w
                y1 = y + dr * cell_h
                x2 = x + (dc + 1) * cell_w
                y2 = y + (dr + 1) * cell_h

                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=HAND_LOW_COLOR, outline=HAND_LOW_COLOR)
                txt = self.canvas.create_text((x1 + x2) * 0.5, (y1 + y2) * 0.5, text="", font=("Arial", 8), fill="white")

                rect_row.append(rect)
                text_row.append(txt)

            self.cell_items[key].append(rect_row)
            self.text_items[key].append(text_row)

        self.canvas.create_text(x + w * 0.5, y + h + 20, text=sensor["label"], font=("Arial", 9), fill="black")

    def draw_colorbar(self):
        x = 815
        y = 90
        w = 22
        h = 200
        steps = 120

        for i in range(steps):
            ratio = 1.0 - float(i) / float(steps - 1)
            color = self.color_from_ratio(ratio)
            y1 = y + i * h / steps
            y2 = y + (i + 1) * h / steps
            self.canvas.create_rectangle(x, y1, x + w, y2, fill=color, outline=color)

        self.canvas.create_text(x - 10, y, text="Max", anchor="e", font=("Arial", 8), fill="black")
        self.canvas.create_text(x - 10, y + h, text="Min", anchor="e", font=("Arial", 8), fill="black")

    def color_from_ratio(self, ratio):
        ratio = clamp(float(ratio), 0.0, 1.0)

        if HAND_COLOR_GAMMA != 1.0:
            ratio = ratio ** HAND_COLOR_GAMMA

        points = [
            (0.00, 75, 69, 165),
            (0.25, 45, 155, 210),
            (0.50, 80, 205, 130),
            (0.75, 220, 230, 35),
            (1.00, 255, 180, 35),
        ]

        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            if ratio >= p0[0] and ratio <= p1[0]:
                t = (ratio - p0[0]) / (p1[0] - p0[0])
                r = int(p0[1] * (1.0 - t) + p1[1] * t)
                g = int(p0[2] * (1.0 - t) + p1[2] * t)
                b = int(p0[3] * (1.0 - t) + p1[3] * t)
                return "#{:02x}{:02x}{:02x}".format(r, g, b)

        return "#ffb423"

    def update_scale(self, values):
        if not self.auto_scale.get():
            self.gui_scale_max = float(HAND_MANUAL_MAX)
            return self.gui_scale_max

        peak = 0.0
        for sensor in HAND_SENSORS:
            key = sensor["key"]
            for row in values.get(key, []):
                for value in row:
                    try:
                        peak = max(peak, float(value))
                    except Exception:
                        pass

        target = max(peak * HAND_AUTO_SCALE_MARGIN, HAND_AUTO_SCALE_FLOOR)
        if target > self.gui_scale_max:
            self.gui_scale_max = target
        else:
            self.gui_scale_max = self.gui_scale_max * HAND_AUTO_SCALE_DECAY + target * (1.0 - HAND_AUTO_SCALE_DECAY)

        if self.gui_scale_max < 1.0:
            self.gui_scale_max = 1.0

        return self.gui_scale_max

    def reset_values(self):
        
        self.gui_scale_max = float(HAND_MANUAL_MAX)

        for sensor in HAND_SENSORS:
            key = sensor["key"]
            rows = sensor["rows"]
            cols = sensor["cols"]

            for r in range(rows):
                for c in range(cols):
                    self.display_values[key][r][c] = 0.0
                    self.canvas.itemconfig(self.cell_items[key][r][c], fill=HAND_LOW_COLOR, outline=HAND_LOW_COLOR)
                    self.canvas.itemconfig(self.text_items[key][r][c], text="")

            self.canvas.itemconfig(self.info_items[key], text="sum: 0\npeak: 0")

        self.status.config(text="state=ready  action=none  ratio=0.000  avg=0.0  max=0.0")

    def update_values(self, values, status_text=""):
        scale_max = self.update_scale(values)
        global_peak = 0.0

        for sensor in HAND_SENSORS:
            key = sensor["key"]
            rows = sensor["rows"]
            cols = sensor["cols"]

            total = 0
            peak = 0

            for r in range(rows):
                for c in range(cols):
                    try:
                        target_value = float(values[key][r][c])
                    except Exception:
                        target_value = 0.0

                    old_value = float(self.display_values[key][r][c])
                    value = old_value * (1.0 - self.smooth_alpha) + target_value * self.smooth_alpha
                    if value < 2.0:
                        value = 0.0
                    self.display_values[key][r][c] = value

                    value_int = int(value)
                    total += value_int
                    peak = max(peak, value_int)
                    global_peak = max(global_peak, value)

                    ratio = value / scale_max if scale_max > 0 else 0.0
                    color = self.color_from_ratio(ratio)

                    self.canvas.itemconfig(self.cell_items[key][r][c], fill=color, outline=color)
                    self.canvas.itemconfig(self.text_items[key][r][c], text="")

            info = "sum: {}\npeak: {}".format(total, peak)
            self.canvas.itemconfig(self.info_items[key], text=info)

        msg = status_text
        if not msg:
            msg = "scale: {}  peak: {}".format(int(scale_max), int(global_peak))
        self.status.config(text=msg)


class AdaptiveGripperUI:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.shared = SharedState()

        self.root.title("Adaptive Gripper FINAL")
        self.root.geometry("1500x800")
        self.root.minsize(1280, 720)
        self.root.configure(bg="#dfe6ee")

        
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<F11>", self.enter_fullscreen)
        self.root.after(200, self.enter_fullscreen)

        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.target_class_var = tk.StringVar(value=args.target_class)
        self.object_instance_var = tk.StringVar(value=getattr(args, "object_instance", ""))
        self.auto_grasp_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Initializing...")
        self.runtime_state_var = tk.StringVar(value="INIT")
        self.last_state_var = tk.StringVar(value="none")
        self.last_action_var = tk.StringVar(value="none")
        self.ratio_var = tk.StringVar(value="0.000")
        self.adjustments_var = tk.StringVar(value="0")
        self.probs_var = tk.StringVar(value="success=0.00 slip=0.00 overforce=0.00")
        self.tactile_summary_var = tk.StringVar(value="total_avg=0 total_max=0 area=0")
        self.log_var = tk.StringVar(value="No logs yet.")
        self._camera_photo = None
        self._last_log_seen = None
        self._last_runtime_state_seen = None

        self.build_ui()

        self.sync_inputs_to_shared()

        self.worker = AdaptiveRuntimeWorker(args=self.args, shared=self.shared)
        self.worker.start()

        self.update_ui_loop()

    def enter_fullscreen(self, event=None):
        try:
            self.root.attributes("-fullscreen", True)
        except Exception:
            pass

        try:
            self.root.attributes("-zoomed", True)
        except Exception:
            pass

        try:
            self.root.state("zoomed")
        except Exception:
            pass

        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            self.root.geometry("{}x{}+0+0".format(sw, sh))
        except Exception:
            pass

    def exit_fullscreen(self, event=None):
        try:
            self.root.attributes("-fullscreen", False)
        except Exception:
            pass

        try:
            self.root.attributes("-zoomed", False)
        except Exception:
            pass

    def build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.configure("Panel.TLabelframe", background="#f4f6f8")
            style.configure("Panel.TLabelframe.Label", font=("Arial", 10, "bold"))
            style.configure("Big.TButton", font=("Arial", 12, "bold"), padding=8)
            style.configure("Small.TLabel", font=("Arial", 9))
        except Exception:
            pass

        main = tk.Frame(self.root, bg="#dfe6ee")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        main.grid_columnconfigure(0, weight=0, minsize=590)
        main.grid_columnconfigure(1, weight=1, minsize=880)
        main.grid_rowconfigure(0, weight=0, minsize=395)
        main.grid_rowconfigure(1, weight=1, minsize=360)

        # Top-left: camera
        camera_panel = ttk.LabelFrame(main, text="Camera / YOLO Detection", style="Panel.TLabelframe")
        camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.camera_label = tk.Label(camera_panel, bg="#111111")
        self.camera_label.pack(fill="both", expand=True, padx=8, pady=8)

        # Bottom-left: terminal/log
        terminal_panel = ttk.LabelFrame(main, text="Terminal / Runtime Log", style="Panel.TLabelframe")
        terminal_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 0))

        self.log_text = tk.Text(
            terminal_panel,
            bg="#101010",
            fg="#d8d8d8",
            insertbackground="#ffffff",
            font=("Consolas", 8),
            wrap="word",
            height=20,
            relief=tk.FLAT,
        )
        self.log_text.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)

        log_scroll = ttk.Scrollbar(terminal_panel, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        # Top-right: controls 
        control_panel = ttk.LabelFrame(main, text="Robot Controls", style="Panel.TLabelframe")
        control_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=(0, 8))
        control_panel.grid_columnconfigure(0, weight=1)

        button_outer = tk.Frame(control_panel, bg="#f4f6f8")
        button_outer.pack(side="top", fill="both", expand=True, padx=18, pady=18)

        button_frame = tk.Frame(button_outer, bg="#f4f6f8")
        button_frame.pack(side="top", fill="x", expand=False)
        for col in range(2):
            button_frame.grid_columnconfigure(col, weight=1)

        ttk.Button(button_frame, text="Start Grasp", style="Big.TButton", command=self.on_start_grasp).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(button_frame, text="Open / Release", style="Big.TButton", command=self.on_open).grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(button_frame, text="Calibrate", style="Big.TButton", command=self.on_calibrate).grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(button_frame, text="Rearm", style="Big.TButton", command=self.on_rearm).grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(button_frame, text="Quit", style="Big.TButton", command=self.on_quit).grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

        auto_frame = tk.Frame(button_outer, bg="#f4f6f8")
        auto_frame.pack(side="top", fill="x", padx=2, pady=(8, 0))
        tk.Checkbutton(
            auto_frame,
            text="Auto grasp",
            variable=self.auto_grasp_var,
            command=self.on_toggle_auto_grasp,
            bg="#f4f6f8",
            fg="#111111",
            activebackground="#f4f6f8",
            selectcolor="#ffffff",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(side="left", anchor="w")

       
        status_grid = tk.Frame(button_outer, bg="#f4f6f8")
        status_grid.pack(side="top", fill="both", expand=True, padx=2, pady=(18, 0))
        for col in range(3):
            status_grid.grid_columnconfigure(col, weight=1)
        for row in range(2):
            status_grid.grid_rowconfigure(row, weight=1)

        self.make_status_card(status_grid, "Runtime", self.runtime_state_var, 0, 0)
        self.make_status_card(status_grid, "Action", self.last_action_var, 0, 1)
        self.make_status_card(status_grid, "Ratio", self.ratio_var, 0, 2)
        self.make_status_card(status_grid, "Model State", self.last_state_var, 1, 0)
        self.make_status_card(status_grid, "Probabilities", self.probs_var, 1, 1)
        self.make_status_card(status_grid, "Adjustments", self.adjustments_var, 1, 2)

       
        heatmap_panel = ttk.LabelFrame(main, text="Hand Display / Tactile Heatmap", style="Panel.TLabelframe")
        heatmap_panel.grid(row=1, column=1, sticky="nsew", padx=(0, 0), pady=(0, 0))

        self.hand_canvas = HandDisplayCanvas(heatmap_panel)
        
        self.hand_canvas.frame.pack(fill="both", expand=True, padx=6, pady=6)

    def make_status_card(self, parent, title, variable, row, col):
        
        frame = tk.Frame(parent, bg="#ffffff", bd=1, relief=tk.SOLID)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        title_label = tk.Label(
            frame,
            text=str(title),
            bg="#ffffff",
            fg="#303030",
            disabledforeground="#303030",
            font=("Arial", 9, "bold"),
            anchor="w",
            padx=8,
            pady=4,
            relief=tk.FLAT,
            highlightthickness=0,
        )
        title_label.grid(row=0, column=0, sticky="ew")

        value_label = tk.Label(
            frame,
            textvariable=variable,
            bg="#ffffff",
            fg="#111111",
            disabledforeground="#111111",
            font=("Arial", 12),
            anchor="w",
            justify="left",
            wraplength=245,
            padx=8,
            pady=4,
            relief=tk.FLAT,
            highlightthickness=0,
        )
        value_label.grid(row=1, column=0, sticky="nsew")

    def append_log(self, message):
       
        if not message:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(message) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def on_toggle_auto_grasp(self):
        self.shared.update_runtime_values(auto_grasp=bool(self.auto_grasp_var.get()))

    def sync_inputs_to_shared(self):
        self.shared.update_runtime_values(
            auto_grasp=bool(self.auto_grasp_var.get()),
            target_class=self.target_class_var.get().strip(),
            object_instance=self.object_instance_var.get().strip(),
        )

    def on_start_grasp(self):
        self.sync_inputs_to_shared()
        self.shared.set_flag("pending_start", True)

    def on_open(self):
        self.sync_inputs_to_shared()
        self.shared.set_flag("pending_open", True)

    def on_calibrate(self):
        self.sync_inputs_to_shared()
        self.shared.set_flag("pending_calibrate", True)

    def on_rearm(self):
        self.sync_inputs_to_shared()
        self.shared.set_flag("pending_rearm", True)

    def on_quit(self):
        self.shared.set_flag("pending_quit", True)
        self.shared.running = False
        self.root.after(250, self.root.destroy)

    def update_camera(self, frame_bgr):
        if frame_bgr is None:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

       
        image.thumbnail((600, 420))
        bg = Image.new("RGB", (600, 420), (15, 15, 15))
        x = (600 - image.width) // 2
        y = (420 - image.height) // 2
        bg.paste(image, (x, y))

        photo = ImageTk.PhotoImage(image=bg)
        self._camera_photo = photo
        self.camera_label.configure(image=photo)

    def update_heatmap(self, tactile_snapshot, snap):
        # When the gripper is opened/reset, return the hand display to the initial
        
        if snap["runtime_state"] == "OPEN" and float(snap["current_ratio"]) <= 0.001:
            if self._last_runtime_state_seen != "OPEN":
                self.hand_canvas.reset_values()
            self._last_runtime_state_seen = "OPEN"
            return

        self._last_runtime_state_seen = snap["runtime_state"]

        status_text = "state={}  action={}  ratio={:.3f}  avg={:.1f}  max={:.1f}".format(
            snap["last_state"],
            snap["last_action"],
            float(snap["current_ratio"]),
            float(snap["total_avg"]),
            float(snap["total_max"]),
        )
        self.hand_canvas.update_values(tactile_snapshot, status_text=status_text)

    def update_ui_loop(self):
        self.sync_inputs_to_shared()
        snap = self.shared.get_snapshot()

        self.runtime_state_var.set(snap["runtime_state"])
        self.last_state_var.set(snap["last_state"])
        self.last_action_var.set(snap["last_action"])
        self.ratio_var.set("{:.3f}".format(float(snap["current_ratio"])))
        self.adjustments_var.set(str(int(snap["total_adjustments"])))
        self.probs_var.set("success={:.2f}  slip={:.2f}  overforce={:.2f}".format(float(snap["prob_success"]), float(snap["prob_slip"]), float(snap["prob_overforce"])))
        self.tactile_summary_var.set("avg={:.1f}  max={:.1f}  area={:.3f}  no={}".format(float(snap["total_avg"]), float(snap["total_max"]), float(snap["contact_area_avg"]), int(snap["no_contact_count"])))
        self.status_var.set(snap["status_line"])
        self.log_var.set(snap["last_log"])

        if snap["last_log"] != self._last_log_seen:
            self._last_log_seen = snap["last_log"]
            self.append_log(snap["last_log"])

        self.update_camera(snap["camera_frame_bgr"])
        self.update_heatmap(snap["tactile_snapshot"], snap)

        if snap["running"]:
            self.root.after(35, self.update_ui_loop)
        else:
            self.root.after(250, self.root.destroy)


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive gripper UI - Step 9")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=IOU_THRESH)
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--infer-every", type=int, default=DEFAULT_INFER_EVERY)
    parser.add_argument("--target-class", default=DEFAULT_TARGET_CLASS)
    parser.add_argument("--object-instance", default="", help="Known object instance, example ball_1/can_2/bottle_3")
    parser.add_argument("--trigger-zone", default=DEFAULT_TRIGGER_ZONE)
    parser.add_argument("--min-box-area-ratio", type=float, default=MIN_BOX_AREA_RATIO)
    parser.add_argument("--stable-count", type=int, default=DETECT_CONSECUTIVE_FRAMES)
    parser.add_argument("--stable-iou", type=float, default=STABLE_IOU_THRESH)
    parser.add_argument("--camera-fourcc", default=CAMERA_FOURCC)
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    parser.add_argument("--camera-buffer", type=int, default=CAMERA_BUFFER)
    parser.add_argument("--window-sec", type=float, default=0.4)
    parser.add_argument("--sample-interval", type=float, default=0.02)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--no-contact-total-max", type=float, default=1.0)
    parser.add_argument("--no-contact-area", type=float, default=0.001)
    parser.add_argument("--no-contact-windows", type=int, default=5)
    parser.add_argument("--max-slip-adjustments", type=int, default=-1)
    parser.add_argument("--max-overforce-decreases", type=int, default=8)
    parser.add_argument("--persistent-slip-windows-after-max", type=int, default=3)
    parser.add_argument("--persistent-overforce-windows-after-max", type=int, default=2)
    parser.add_argument("--holding-camera-sleep", type=float, default=0.10)

    # Final version CSV 
    parser.add_argument("--event-log-dir", default="./event_logs", help="Directory for event log CSV files.")
    parser.add_argument("--event-log-file", default="", help="Exact CSV path. If empty, a timestamped file is created in --event-log-dir.")
    parser.add_argument("--disable-event-log", action="store_true", help="Disable event log CSV output.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    AdaptiveGripperUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()


