from model.segmentation_model import SegmentationModel
from utils.preprocessing import preprocess_frame, validate_frame


def evaluate():
    print("=" * 40)
    print("Evaluating Model")
    print("=" * 40)

    model = SegmentationModel()
    model.load_weights()

    test_frames = ["test1", "test2", "test3"]

    for frame in test_frames:
        print("\nFrame:", frame)

        f = preprocess_frame(frame)

        if not validate_frame(f):
            continue

        preds = model.predict(f)
        filtered = model.filter_predictions(preds)

        for p in filtered:
            print(p["label"], "|", p["confidence"])

    print("\nEvaluation finished")


if __name__ == "__main__":
    evaluate()
