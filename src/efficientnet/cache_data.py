"""Create the exact patient-grouped split and resized image cache."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

CSV_PATH = Path("data/train.csv")
IMAGE_DIR = Path("data/jpeg/train")
CACHE_DIR = Path("data/cache")
SPLIT_DIR = Path("data/splits")
IMAGE_SIZE = 224
SEED = 42

EXPECTED = {
    "train": (23043, 22644, 399),
    "validation": (4903, 4830, 73),
    "test": (5180, 5068, 112),
}

def make_splits():
    df = pd.read_csv(CSV_PATH)
    required = {"image_name", "patient_id", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(splitter.split(df, groups=df["patient_id"]))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(splitter.split(temp_df, groups=temp_df["patient_id"]))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    splits = {"train": train_df, "validation": val_df, "test": test_df}

    patient_sets = [set(part.patient_id) for part in splits.values()]
    assert not (patient_sets[0] & patient_sets[1])
    assert not (patient_sets[0] & patient_sets[2])
    assert not (patient_sets[1] & patient_sets[2])

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        counts = (
            len(part),
            int((part.target == 0).sum()),
            int((part.target == 1).sum()),
        )
        print(f"{name}: total={counts[0]}, class0={counts[1]}, class1={counts[2]}")
        if counts != EXPECTED[name]:
            raise ValueError(
                f"{name} split mismatch. Expected {EXPECTED[name]}, got {counts}. "
                "Check that you are using the same train.csv as the Kaggle run."
            )

        part["source_path"] = part["image_name"].map(
            lambda name: str(IMAGE_DIR / f"{name}.jpg")
        )
        part["image_path"] = part["image_name"].map(
            lambda name: str(CACHE_DIR / name[:1] / f"{name}.jpg")
        )

        missing = [p for p in part.source_path if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing source image: {missing[0]}")

        part.to_csv(SPLIT_DIR / f"{name}.csv", index=False)

    return splits

def cache_one(task):
    source, destination = task
    source, destination = Path(source), Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        ).save(destination, format="JPEG", quality=95)

def main():
    splits = make_splits()
    tasks = []
    for part in splits.values():
        for row in part.itertuples():
            tasks.append((row.source_path, row.image_path))

    print(f"\nCaching {len(tasks):,} images...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(cache_one, tasks))
    print(f"Cache complete: {CACHE_DIR.resolve()}")
    print(f"Split CSVs saved in: {SPLIT_DIR.resolve()}")

if __name__ == "__main__":
    main()
