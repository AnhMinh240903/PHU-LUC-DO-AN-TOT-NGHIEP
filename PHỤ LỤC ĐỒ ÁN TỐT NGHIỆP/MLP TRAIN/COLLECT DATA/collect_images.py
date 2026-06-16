#!/home/anhminh/yolo/bin/python3


import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
import threading

import cv2
import numpy as np
from ultralytics import YOLO



class FrameGrabber:
    

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = -1
        self.running = False
        self.thread = None

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

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return None, -1
            return self.frame.copy(), self.frame_id

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class StableTrigger:
    

    def __init__(self, required_count=3, iou_thresh=0.25):
        self.required_count = int(max(1, required_count))
        self.iou_thresh = float(max(0.0, min(1.0, iou_thresh)))
        self.prev = None
        self.count = 0

    @staticmethod
    def bbox_iou(a, b):
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

    def update(self, candidate):
        if candidate is None:
            self.prev = None
            self.count = 0
            return None
        if self.prev is None:
            self.count = 1
        else:
            same_class = candidate['class_name'] == self.prev['class_name']
            iou = self.bbox_iou(candidate['xyxy'], self.prev['xyxy'])
            if same_class and iou >= self.iou_thresh:
                self.count += 1
            else:
                self.count = 1
        self.prev = candidate
        if self.count >= self.required_count:
            return candidate
        return None




def parse_resolution(text: str):
    
    try:
        w_str, h_str = text.lower().split('x')
        return int(w_str), int(h_str)
    except Exception as exc:
        raise ValueError(f"Invalid resolution '{text}'. Use WIDTHxHEIGHT.") from exc


def parse_zone(text: str):
    
    vals = [float(v.strip()) for v in text.split(',')]
    x1, y1, x2, y2 = vals
    return (max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2)))


def zone_to_pixels(zone, width, height):
    x1, y1, x2, y2 = zone
    return (int(round(x1 * width)), int(round(y1 * height)),
            int(round(x2 * width)), int(round(y2 * height)))


def draw_detection(frame, xyxy, class_name, conf, color=(0, 255, 0)):
  
    xmin, ymin, xmax, ymax = xyxy
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    label = f"{class_name}: {int(conf * 100)}%"
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_ymin = max(ymin, label_size[1] + 10)
    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                  (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
    cv2.putText(frame, label, (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def run_inference(model, frame, imgsz, conf, iou, max_det):
   
    return model.predict(source=frame, imgsz=imgsz, conf=conf,
                         iou=iou, max_det=max_det, verbose=False)


def select_best_candidate(results, labels, global_thresh, target_class,
                          frame_shape, min_box_area_ratio, zone_px):
  
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
        class_name = labels[class_idx]
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
        candidates.append({
            'conf': conf, 'class_idx': class_idx, 'class_name': class_name,
            'xyxy': xyxy, 'area': area, 'center': (cx, cy)
        })
    if not candidates:
        return None
    candidates.sort(key=lambda d: (d['conf'], d['area']), reverse=True)
    return candidates[0]


def open_camera(source: str, width: int = None, height: int = None,
                fourcc: str = 'MJPG', fps: int = 30, buffer_size: int = 1):
    
    if source.startswith('usb'):
        cam_index = int(source[3:])
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    elif source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if buffer_size:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap






def capture_loop(model, grabber: FrameGrabber, labels, args):
    
    os.makedirs(args.output_dir, exist_ok=True)
    labels_path = os.path.join(args.output_dir, 'labels.csv')
    with open(labels_path, 'a', newline='') as lf:
       
        fieldnames = [
            'image', 'object_class', 'object_instance',
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'bbox_w_norm', 'bbox_h_norm', 'bbox_area_norm',
            'aspect_ratio', 'cx_norm', 'cy_norm', 'conf'
        ]
        writer = csv.DictWriter(lf, fieldnames=fieldnames)
        if lf.tell() == 0:
            writer.writeheader()
        captured = 0
       
        stable_filter = StableTrigger(required_count=args.stable_count,
                                      iou_thresh=args.stable_iou)
       
        try:
            res_w, res_h = parse_resolution(args.resolution)
        except Exception:
            res_w, res_h = 640, 480
        zone_vals = parse_zone(args.trigger_zone)
        zone_px = zone_to_pixels(zone_vals, res_w, res_h)
       
        last_seen_frame_id = -1
        last_infer_frame_id = -1
        new_frame_counter = 0
        best_candidate = None
        stable_candidate = None
        while True:
            frame, frame_id = grabber.read_latest()
            if frame is None:
                time.sleep(0.01)
                continue
           
            if frame_id != last_seen_frame_id:
                last_seen_frame_id = frame_id
                new_frame_counter += 1
            
            if frame.shape[1] != res_w or frame.shape[0] != res_h:
                frame = cv2.resize(frame, (res_w, res_h))
            display_frame = frame.copy()
           
            should_infer = (
                frame_id != last_infer_frame_id and
                (new_frame_counter % max(1, args.infer_every) == 0)
            )
            if should_infer:
                results = run_inference(model, frame, args.imgsz,
                                        args.thresh, args.iou, args.max_det)
                last_infer_frame_id = frame_id
                best_candidate = select_best_candidate(
                    results, labels, args.thresh, args.object_class,
                    frame.shape, args.min_box_area_ratio, zone_px)
                stable_candidate = stable_filter.update(best_candidate)
           
            if best_candidate is not None:
                color = (0, 255, 0)
                draw_detection(display_frame, best_candidate['xyxy'],
                               best_candidate['class_name'],
                               best_candidate['conf'], color)
            
            cv2.rectangle(display_frame,
                          (zone_px[0], zone_px[1]),
                          (zone_px[2], zone_px[3]),
                          (0, 255, 255), 2)
            cv2.putText(display_frame,
                        f"Captured: {captured}/{args.count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.imshow('Collect Images', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('[INFO] Early exit requested.')
                break
           
            if stable_candidate is not None and key == ord('c'):
                xyxy = stable_candidate['xyxy']
                xmin, ymin, xmax, ymax = xyxy
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(frame.shape[1] - 1, xmax)
                ymax = min(frame.shape[0] - 1, ymax)
                if ymax > ymin and xmax > xmin:
                    crop = frame[ymin:ymax, xmin:xmax]
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    filename = (
                        f"{args.object_class}_{args.object_id}_"
                        f"{timestamp}_{captured:04d}.jpg"
                    )
                    filepath = os.path.join(args.output_dir, filename)
                    cv2.imwrite(filepath, crop)
                    
                    frame_h, frame_w = frame.shape[:2]
                    w = xmax - xmin
                    h = ymax - ymin
                    area = float(w * h)
                    bbox_w_norm = w / float(frame_w)
                    bbox_h_norm = h / float(frame_h)
                    bbox_area_norm = area / float(frame_w * frame_h)
                    aspect_ratio = (float(w) / float(h)) if h > 0 else 0.0
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    cx_norm = cx / float(frame_w)
                    cy_norm = cy / float(frame_h)
                    conf = float(stable_candidate['conf'])
                    label_row = {
                        'image': filename,
                        'object_class': args.object_class,
                        'object_instance': args.object_id,
                        'bbox_x1': xmin,
                        'bbox_y1': ymin,
                        'bbox_x2': xmax,
                        'bbox_y2': ymax,
                        'bbox_w_norm': bbox_w_norm,
                        'bbox_h_norm': bbox_h_norm,
                        'bbox_area_norm': bbox_area_norm,
                        'aspect_ratio': aspect_ratio,
                        'cx_norm': cx_norm,
                        'cy_norm': cy_norm,
                        'conf': conf,
                    }
                    writer.writerow(label_row)
                    lf.flush()
                    print(f"[INFO] Saved {filepath}")
                    captured += 1
                    
                    stable_filter.update(None)
                    best_candidate = None
                    stable_candidate = None
                    if captured >= args.count:
                        break
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Collect object images for grasp dataset')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the YOLO model directory or file')
    parser.add_argument('--source', type=str, default='usb0',
                        help='Video source (e.g. usb0, 0, or a video file path)')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Camera resolution, e.g. 640x480')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='YOLO inference image size')
    parser.add_argument('--thresh', type=float, default=0.25,
                        help='YOLO objectness threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='YOLO IoU threshold')
    parser.add_argument('--max-det', type=int, default=12,
                        help='Maximum number of detections per image')
    parser.add_argument('--min-box-area-ratio', type=float, default=0.012,
                        help='Minimum bounding box area relative to frame area')
    parser.add_argument('--trigger-zone', type=str, default='0.22,0.18,0.78,0.92',
                        help='Trigger zone in normalised coordinates x1,y1,x2,y2')
    parser.add_argument('--stable-count', type=int, default=3,
                        help='Number of consecutive frames for a stable detection')
    parser.add_argument('--stable-iou', type=float, default=0.25,
                        help='Minimum IoU between consecutive detections to be considered the same object')
    parser.add_argument('--infer-every', type=int, default=1,
                        help='Run inference every N frames (1 = every frame)')
    parser.add_argument('--object-class', type=str, required=True,
                        help='Target class name (e.g. ball, can, bottle)')
    parser.add_argument('--object-id', type=str, required=True,
                        help='Specific instance identifier (e.g. ball_1)')
    parser.add_argument('--count', type=int, default=50,
                        help='Number of images to capture')
    parser.add_argument('--output-dir', type=str, default='dataset/images',
                        help='Directory where images and labels.csv will be stored')
    args = parser.parse_args()

   
    try:
        res_w, res_h = map(int, args.resolution.split('x'))
    except Exception:
        print('ERROR: invalid resolution format. Use WIDTHxHEIGHT, e.g. 640x480.')
        sys.exit(1)

    # Load YOLO model
    print('[INFO] Loading YOLO model...')
    model = YOLO(args.model, task='detect')
    labels = model.names

    # Open camera
    print('[INFO] Opening camera...')
    cap = open_camera(args.source, res_w, res_h)
    if not cap or not cap.isOpened():
        print('ERROR: Failed to open video source.')
        sys.exit(1)
    grabber = FrameGrabber(cap).start()

    try:
        capture_loop(model, grabber, labels, args)
    finally:
        grabber.stop()
        if cap:
            cap.release()


if __name__ == '__main__':
  
    import threading 
    main()

