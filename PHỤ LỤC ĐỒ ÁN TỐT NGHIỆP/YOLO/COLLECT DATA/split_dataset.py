import random
import shutil
from pathlib import Path


DATASET_DIR = Path(".") 
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


GROUP_PREFIXES = {
    "b": "ball",
    "c": "can",
    "l": "bottle",
}


def find_images(images_dir: Path):
    files = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def group_by_prefix(image_files):
    groups = {k: [] for k in GROUP_PREFIXES.keys()}
    unknown = []

    for img_path in image_files:
        stem = img_path.stem.lower()
        matched = False
        for prefix in GROUP_PREFIXES.keys():
            if stem.startswith(prefix):
                groups[prefix].append(img_path)
                matched = True
                break
        if not matched:
            unknown.append(img_path)

    return groups, unknown


def copy_pair(img_path: Path, src_labels_dir: Path, dst_images_dir: Path, dst_labels_dir: Path):
    label_path = src_labels_dir / f"{img_path.stem}.txt"

    if not label_path.exists():
        print(f"[WARNING] Missing label for image: {img_path.name}")
        return False

    shutil.copy2(img_path, dst_images_dir / img_path.name)
    shutil.copy2(label_path, dst_labels_dir / label_path.name)
    return True


def main():
    random.seed(RANDOM_SEED)

    src_images_dir = DATASET_DIR / "images"
    src_labels_dir = DATASET_DIR / "labels"

    train_images_dir = src_images_dir / "train"
    val_images_dir = src_images_dir / "val"
    train_labels_dir = src_labels_dir / "train"
    val_labels_dir = src_labels_dir / "val"

  
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

   
    image_files = [
        p for p in find_images(src_images_dir)
        if p.parent == src_images_dir
    ]

    groups, unknown = group_by_prefix(image_files)

    if unknown:
        print("\n[WARNING] Các file không khớp prefix b/c/l:")
        for p in unknown:
            print(" -", p.name)

    total_train = 0
    total_val = 0

    print("===== SPLIT SUMMARY =====")
    for prefix, files in groups.items():
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = n_total - n_train

        train_files = files[:n_train]
        val_files = files[n_train:]

        copied_train = 0
        copied_val = 0

        for img_path in train_files:
            ok = copy_pair(img_path, src_labels_dir, train_images_dir, train_labels_dir)
            if ok:
                copied_train += 1

        for img_path in val_files:
            ok = copy_pair(img_path, src_labels_dir, val_images_dir, val_labels_dir)
            if ok:
                copied_val += 1

        total_train += copied_train
        total_val += copied_val

        print(
            f"{GROUP_PREFIXES[prefix]:<8} | total={n_total:<3} "
            f"train={copied_train:<3} val={copied_val:<3}"
        )

    print("\n===== TOTAL =====")
    print(f"train = {total_train}")
    print(f"val   = {total_val}")
    print(f"total = {total_train + total_val}")


if __name__ == "__main__":
    main()