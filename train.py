from model.segmentation_model import SegmentationModel
from utils.preprocessing import preprocess_frame, augment_frame, validate_frame


def load_dataset():
    print("Loading training dataset...")

    return [
        "frame_001",
        "frame_002",
        "frame_003",
        "frame_004",
        "frame_005"
    ]


def train_epoch(model, dataset, epoch):
    print(f"\nEpoch {epoch}")

    total = 0

    for frame_name in dataset:
        print("\nFrame:", frame_name)

        frame = preprocess_frame(frame_name)

        if not validate_frame(frame):
            continue

        augmented = augment_frame(frame)

        for name, f in augmented.items():
            print("Augmented:", name)

            preds = model.predict(f)
            filtered = model.filter_predictions(preds)

            total += len(filtered)

            for p in filtered:
                print(p["label"], "|", p["confidence"])

    return total


def train():
    print("=" * 40)
    print("Training Segmentation Model")
    print("=" * 40)

    model = SegmentationModel()
    model.load_weights()

    dataset = load_dataset()

    for epoch in range(1, 6):
        total = train_epoch(model, dataset, epoch)
        print(f"Epoch {epoch} total detections:", total)

    print("Training complete")


if __name__ == "__main__":
    train()
