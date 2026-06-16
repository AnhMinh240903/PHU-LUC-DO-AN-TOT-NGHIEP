import cv2
import os
import time
import argparse
from pathlib import Path


def get_next_index(images_dir: Path, prefix: str) -> int:
    max_idx = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in images_dir.glob(ext):
            stem = p.stem
            if stem.startswith(prefix):
                num_part = stem[len(prefix):]
                if num_part.isdigit():
                    max_idx = max(max_idx, int(num_part))
    return max_idx + 1


def save_yolo_label(label_path: Path, class_id: int, img_w: int, img_h: int, box):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")


def draw_ui(frame, box, session_name, prefix, next_idx, burst_mode, shots_left, next_shot_in, total_captured):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

    info_lines = [
        f"Session: {session_name}",
        f"Prefix: {prefix}",
        f"Next image: {prefix}{next_idx}",
        f"Total captured: {total_captured}",
        "Press 1: capture 10 images, 1 image / second",
        "Press 2: exit"
    ]

    if burst_mode:
        info_lines.append(f"Capturing... remaining: {shots_left}")
        info_lines.append(f"Next shot in: {max(0.0, next_shot_in):.1f}s")

    y = 30
    for line in info_lines:
        cv2.putText(
            frame, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
        )
        y += 30

    cv2.putText(
        frame,
        "Place object inside the green box",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--session-name", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--burst-count", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    images_dir = Path("images")
    labels_dir = Path("labels")
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open webcam. Try --camera 0 or --camera 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    next_index = get_next_index(images_dir, args.prefix)
    total_captured = next_index - 1

    burst_mode = False
    shots_left = 0
    last_capture_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from webcam.")
            break

        img_h, img_w = frame.shape[:2]

        box_w = int(img_w * 0.45)
        box_h = int(img_h * 0.60)

        x1 = (img_w - box_w) // 2
        y1 = (img_h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        box = (x1, y1, x2, y2)

        now = time.time()
        next_shot_in = 0.0

        if burst_mode:
            elapsed = now - last_capture_time
            next_shot_in = args.interval - elapsed

            if elapsed >= args.interval:
                img_name = f"{args.prefix}{next_index}.jpg"
                txt_name = f"{args.prefix}{next_index}.txt"

                img_path = images_dir / img_name
                txt_path = labels_dir / txt_name

                cv2.imwrite(str(img_path), frame)
                save_yolo_label(txt_path, args.class_id, img_w, img_h, box)

                print(f"Saved: {img_path} | {txt_path}")

                next_index += 1
                total_captured += 1
                shots_left -= 1
                last_capture_time = now

                if shots_left <= 0:
                    burst_mode = False
                    print("Done capturing 10 images.")

        display = frame.copy()
        draw_ui(
            display,
            box,
            args.session_name,
            args.prefix,
            next_index,
            burst_mode,
            shots_left,
            next_shot_in,
            total_captured
        )

        cv2.imshow("YOLO Data Capture", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("1"):
            if not burst_mode:
                burst_mode = True
                shots_left = args.burst_count
                last_capture_time = time.time() - args.interval
                print(f"Start capturing {args.burst_count} images for {args.session_name}...")
        elif key == ord("2"):
            print("Exit program.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    