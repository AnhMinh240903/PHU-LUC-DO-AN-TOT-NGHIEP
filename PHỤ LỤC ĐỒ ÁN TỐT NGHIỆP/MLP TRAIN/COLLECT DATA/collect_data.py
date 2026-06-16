#!/home/anhminh/yolo/bin/python3


import argparse
import csv
import glob
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import json

try:
    import pigpio  # noqa: F401
except Exception:
    pigpio = None

from tactile_sensor import TactileSensor, SENSOR_CONFIG




RIGHT_PIN = 17
LEFT_PIN = 18
CALIBRATION_FILE = "servo_calibration.json"
P_MIN = 500
P_MAX = 2500


class GripperPigpio:
   

    def __init__(self, hold_refresh_offset_deg: int = 1,
                 hold_refresh_interval_sec: float = 0.12,
                 hold_refresh_pulse_sec: float = 0.04,
                 open_prebias_deg: int = 18):
        if pigpio is None:
            raise RuntimeError('pigpio Python module not found. Ensure pigpiod is running.')
        
        try:
            with open(CALIBRATION_FILE, "r") as f:
                calib = json.load(f)
            self.right_open = int(calib[str(RIGHT_PIN)]["open"])
            self.right_close = int(calib[str(RIGHT_PIN)]["close"])
            self.left_open = int(calib[str(LEFT_PIN)]["open"])
            self.left_close = int(calib[str(LEFT_PIN)]["close"])
            print(f"[INFO] Loaded servo calibration from {CALIBRATION_FILE}")
        except Exception:
            print(f"[WARNING] Cannot load {CALIBRATION_FILE}; using default angles.")
            self.right_open, self.right_close = 30, 90
            self.left_open, self.left_close = 115, 55
        
        self.hold_refresh_interval_sec = float(max(0.05, hold_refresh_interval_sec))
        self.hold_refresh_pulse_sec = float(max(0.02, hold_refresh_pulse_sec))
        self.open_prebias_deg = int(max(0, open_prebias_deg))
       
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError('Cannot connect to pigpiod. Run: sudo pigpiod')
        self.pi.set_mode(RIGHT_PIN, pigpio.OUTPUT)
        self.pi.set_mode(LEFT_PIN, pigpio.OUTPUT)
       
        self.right_open_pulse = self.angle_to_pulse(self.right_open)
        self.right_close_pulse = self.angle_to_pulse(self.right_close)
        self.left_open_pulse = self.angle_to_pulse(self.left_open)
        self.left_close_pulse = self.angle_to_pulse(self.left_close)
       
        self.current_right_pulse = self.right_open_pulse
        self.current_left_pulse = self.left_open_pulse
        self.closed_refresh_active = False
        self.next_refresh_time = 0.0
        self.is_closed = False

    @staticmethod
    def angle_to_pulse(angle: int) -> int:
       
        angle = max(0, min(180, angle))
        ratio = angle / 180.0
        return int(round(P_MIN + ratio * (P_MAX - P_MIN)))

    def set_pulse(self, pin: int, pulse: int):
        pulse = max(P_MIN, min(P_MAX, int(pulse)))
        self.pi.set_servo_pulsewidth(pin, pulse)

    def move_both(self, target_right_pulse: int, target_left_pulse: int):
        
        steps = 70
        start_right = self.current_right_pulse
        start_left = self.current_left_pulse
        for i in range(steps + 1):
            progress = i / steps
            ease = 0.5 - 0.5 * math.cos(math.pi * progress)
            right_pulse = start_right + (target_right_pulse - start_right) * ease
            left_pulse = start_left + (target_left_pulse - start_left) * ease
            self.set_pulse(RIGHT_PIN, right_pulse)
            self.set_pulse(LEFT_PIN, left_pulse)
            delay = 0.018 + (0.055 - 0.018) * (1 - ease)
            time.sleep(delay)
        self.set_pulse(RIGHT_PIN, target_right_pulse)
        self.set_pulse(LEFT_PIN, target_left_pulse)
        self.current_right_pulse = target_right_pulse
        self.current_left_pulse = target_left_pulse

    def release(self):
        self.pi.set_servo_pulsewidth(RIGHT_PIN, 0)
        self.pi.set_servo_pulsewidth(LEFT_PIN, 0)

    def open(self):
       
        self.move_both(self.right_open_pulse, self.left_open_pulse)
        self.is_closed = False
        self.closed_refresh_active = False
        self.release()

    def close_percentage(self, percentage: float):
       
        percentage = max(0.0, min(1.0, percentage))
        target_right = self.right_open_pulse + (self.right_close_pulse - self.right_open_pulse) * percentage
        target_left = self.left_open_pulse + (self.left_close_pulse - self.left_open_pulse) * percentage
        self.move_both(int(target_right), int(target_left))
       
        self.is_closed = True
        self.closed_refresh_active = True
        self.next_refresh_time = time.monotonic() + self.hold_refresh_interval_sec
        self.release()

    def tick_closed_hold(self):
       
        if not self.closed_refresh_active or not self.is_closed:
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

    @staticmethod
    def pulse_to_angle(pulse: float) -> float:
        
        pulse = max(P_MIN, min(P_MAX, float(pulse)))
        ratio = (pulse - P_MIN) / float(P_MAX - P_MIN)
        return ratio * 180.0

    def current_pulses(self):
        
        return self.current_right_pulse, self.current_left_pulse


import math




def compute_features(sequence: List[Dict[str, List[List[float]]]]) -> Dict[str, float]:
   
    features: Dict[str, float] = {}
    total_avg_accum = 0.0
    total_max_val = 0.0
   
    left_sum_total = 0.0
    right_sum_total = 0.0
    contact_sum = 0.0
    total_cells = sum(cfg['rows'] * cfg['cols'] for cfg in SENSOR_CONFIG)
   
    left_keys = {'L_OUT', 'L_IN'}
    right_keys = {'R_OUT', 'R_IN'}
    num_timesteps = len(sequence)
    if num_timesteps == 0:
        
        for cfg in SENSOR_CONFIG:
            features[f"{cfg['key']}_avg"] = 0.0
            features[f"{cfg['key']}_max"] = 0.0
        features['total_avg'] = 0.0
        features['total_max'] = 0.0
        features['left_sum_avg'] = 0.0
        features['right_sum_avg'] = 0.0
        features['contact_area_avg'] = 0.0
        return features
    for cfg in SENSOR_CONFIG:
        key = cfg['key']
        rows, cols = cfg['rows'], cfg['cols']
        sum_val = 0.0
        count = 0
        max_val = 0.0
        for reading in sequence:
            mat = reading[key]
           
            for r in range(rows):
                for c in range(cols):
                    v = mat[r][c]
                    sum_val += v
                    count += 1
                    if v > max_val:
                        max_val = v
            
        avg_val = sum_val / float(count) if count > 0 else 0.0
        features[f'{key}_avg'] = avg_val
        features[f'{key}_max'] = max_val
        total_avg_accum += avg_val
        if max_val > total_max_val:
            total_max_val = max_val
   
    for reading in sequence:
        
        left_sum_step = 0.0
        right_sum_step = 0.0
        active_cells = 0
        for cfg in SENSOR_CONFIG:
            key = cfg['key']
            rows, cols = cfg['rows'], cfg['cols']
            mat = reading[key]
            for r in range(rows):
                for c in range(cols):
                    v = mat[r][c]
                    if v > 0.0:
                        active_cells += 1
                    if key in left_keys:
                        left_sum_step += v
                    elif key in right_keys:
                        right_sum_step += v
        left_sum_total += left_sum_step
        right_sum_total += right_sum_step
        if total_cells > 0:
            contact_sum += active_cells / float(total_cells)
    
    features['total_avg'] = total_avg_accum / float(len(SENSOR_CONFIG))
    features['total_max'] = total_max_val
    features['left_sum_avg'] = left_sum_total / float(num_timesteps) if num_timesteps > 0 else 0.0
    features['right_sum_avg'] = right_sum_total / float(num_timesteps) if num_timesteps > 0 else 0.0
    features['contact_area_avg'] = contact_sum / float(num_timesteps)
    return features





def run_trial_for_image(image_path: str, sensor: TactileSensor, gripper: GripperPigpio,
                        csv_attempts_writer, csv_dataset_writer, attempt_id_start: int,
                        bbox_info: Dict[str, Dict[str, str]]) -> int:
   
    
    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    parts = name.split('_')
    if len(parts) < 2:
        obj_class = 'unknown'
        obj_instance = 'unknown'
    else:
        obj_class = parts[0]
        obj_instance = parts[1]
    print(f"\n=== Starting trial for {basename} (class={obj_class}, instance={obj_instance}) ===")
    print("Place the object in the gripper. You will be asked to enter grip percentages.")
    attempt_id = attempt_id_start
    success = False
    while not success:
        try:
            ratio_str = input("Enter grip percentage (0‑100) or 'q' to skip this object: ").strip()
        except EOFError:
            ratio_str = 'q'
        if ratio_str.lower() == 'q':
            print("[INFO] Skipping this object.")
            break
        try:
            ratio = float(ratio_str) / 100.0
        except Exception:
            print("Invalid input. Please enter a number between 0 and 100.")
            continue
        ratio = max(0.0, min(1.0, ratio))
        print(f"Closing gripper to {ratio*100:.1f}%...")
        
        gripper.close_percentage(ratio)
       
        right_pulse, left_pulse = gripper.current_pulses()
        final_servo_pulse = (right_pulse + left_pulse) / 2.0
        final_servo_angle = (GripperPigpio.pulse_to_angle(right_pulse) + GripperPigpio.pulse_to_angle(left_pulse)) / 2.0
       
        target_force = final_servo_angle
       
        sequence: List[Dict[str, List[List[float]]]] = []
        start_time = time.time()
        print("Sampling tactile sensors... Press one of the following keys when ready:")
        print("  y = success, s = slip, o = overforce, r = retry, f = fail")
        outcome_label = None
        success_flag = 0
        while True:
            forces = sensor.read_force()
            sequence.append(forces)
            gripper.tick_closed_hold()
            if sys.platform == 'linux' and os.isatty(sys.stdin.fileno()):
                import select
                dr, _, _ = select.select([sys.stdin], [], [], 0)
                if dr:
                    line = sys.stdin.readline().strip().lower()
                    if line == 'y':
                        success_flag = 1
                        outcome_label = 'success'
                        success = True
                        print("[INFO] Success recorded.")
                        break
                    elif line == 's':
                        success_flag = 0
                        outcome_label = 'slip'
                        print("[INFO] Slip recorded.")
                        break
                    elif line == 'o':
                        success_flag = 0
                        outcome_label = 'overforce'
                        print("[INFO] Overforce recorded.")
                        break
                    elif line == 'r':
                        success_flag = 0
                        outcome_label = 'retry'
                        print("[INFO] Retry recorded.")
                        break
                    elif line == 'f':
                        success_flag = 0
                        outcome_label = 'fail'
                        print("[INFO] Fail recorded.")
                        break
            time.sleep(0.02)
       
        gripper.open()
        duration = time.time() - start_time
        if not sequence:
            continue
        if outcome_label is None:
           
            continue
        
        attempt_row = {
            'attempt_id': attempt_id,
            'image': basename,
            'object_class': obj_class,
            'object_instance': obj_instance,
            'target_force': target_force,
            'grip_ratio': ratio,
            'success': success_flag,
            'label': outcome_label,
            'duration': duration,
            'final_servo_angle': final_servo_angle,
            'final_servo_pulse': final_servo_pulse,
            'notes': '',
        }
        csv_attempts_writer.writerow(attempt_row)
        
        features = compute_features(sequence)
        dataset_row = {
            'attempt_id': attempt_id,
            'image': basename,
            'object_class': obj_class,
            'object_instance': obj_instance,
            'target_force': target_force,
            'grip_ratio': ratio,
            'final_servo_angle': final_servo_angle,
            'final_servo_pulse': final_servo_pulse,
            'success': success_flag,
            'label': outcome_label,
        }
        
        bbox = bbox_info.get(basename, {})
        for k in ['bbox_w_norm', 'bbox_h_norm', 'bbox_area_norm', 'aspect_ratio',
                  'cx_norm', 'cy_norm', 'conf']:
            if k in bbox:
                dataset_row[k] = float(bbox[k]) if bbox[k] != '' else 0.0
            else:
                dataset_row[k] = 0.0
        
        dataset_row.update(features)
        csv_dataset_writer.writerow(dataset_row)
        attempt_id += 1
        if not success:
            print("Prepare for the next attempt with a larger grip percentage.")
    return attempt_id


def main():
    parser = argparse.ArgumentParser(description='Collect tactile force data for grasping dataset')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing object images captured earlier')
    parser.add_argument('--output-dir', type=str, default='dataset/dataset_out',
                        help='Directory where CSV output files will be stored')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip tactile sensor calibration at startup')
    args = parser.parse_args()
   
    images = sorted(glob.glob(os.path.join(args.images_dir, '*.jpg')) +
                    glob.glob(os.path.join(args.images_dir, '*.png')))
    if not images:
        print(f"ERROR: No images found in {args.images_dir}.")
        sys.exit(1)
    
    labels_csv = os.path.join(args.images_dir, 'labels.csv')
    bbox_info: Dict[str, Dict[str, str]] = {}
    if os.path.exists(labels_csv):
        try:
            with open(labels_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                   
                    bbox_info[row['image']] = row
            print(f"[INFO] Loaded bounding box metadata from {labels_csv}")
        except Exception as e:
            print(f"[WARNING] Failed to read {labels_csv}: {e}")
    os.makedirs(args.output_dir, exist_ok=True)
    attempts_path = os.path.join(args.output_dir, 'grasp_attempts.csv')
    dataset_path = os.path.join(args.output_dir, 'grasp_dataset.csv')
    
    print('[INFO] Initialising tactile sensors...')
    sensor = TactileSensor()
    if not args.skip_calibration:
        print('[INFO] Calibrating sensors (please ensure gripper is open and sensors are free)...')
        sensor.calibrate()
        print('[INFO] Calibration complete.')
   
    print('[INFO] Initialising gripper...')
    try:
        gripper = GripperPigpio()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Make sure pigpiod is running and calibration file exists if required.")
        sensor.cleanup()
        sys.exit(1)
    gripper.open()
    
    fieldnames_attempts = [
        'attempt_id', 'image', 'object_class', 'object_instance',
        'target_force', 'grip_ratio', 'success', 'label', 'duration',
        'final_servo_angle', 'final_servo_pulse', 'notes'
    ]
    
    fieldnames_dataset = (
        ['attempt_id', 'image', 'object_class', 'object_instance',
         'target_force', 'grip_ratio', 'final_servo_angle', 'final_servo_pulse',
         'success', 'label',
         'bbox_w_norm', 'bbox_h_norm', 'bbox_area_norm', 'aspect_ratio',
         'cx_norm', 'cy_norm', 'conf',
         'left_sum_avg', 'right_sum_avg', 'contact_area_avg']
        + [f'{cfg["key"]}_avg' for cfg in SENSOR_CONFIG]
        + [f'{cfg["key"]}_max' for cfg in SENSOR_CONFIG]
        + ['total_avg', 'total_max']
    )
   
    next_attempt_id = 0
    if os.path.exists(attempts_path):
        with open(attempts_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    aid = int(row['attempt_id'])
                    next_attempt_id = max(next_attempt_id, aid + 1)
                except Exception:
                    continue
    with open(attempts_path, 'a', newline='') as f_attempts, \
         open(dataset_path, 'a', newline='') as f_dataset:
        writer_attempts = csv.DictWriter(f_attempts, fieldnames=fieldnames_attempts)
        writer_dataset = csv.DictWriter(f_dataset, fieldnames=fieldnames_dataset)
       
        if f_attempts.tell() == 0:
            writer_attempts.writeheader()
        if f_dataset.tell() == 0:
            writer_dataset.writeheader()
  
        attempt_id = next_attempt_id
        for img_path in images:
            attempt_id = run_trial_for_image(img_path, sensor, gripper,
                                             writer_attempts, writer_dataset,
                                             attempt_id, bbox_info)
    # Clean up
    gripper.cleanup()
    sensor.cleanup()
    print('[INFO] Data collection complete. Files saved to', args.output_dir)


if __name__ == '__main__':
    main()

