from model.segmentation_model import SegmentationModel
from utils.preprocessing import preprocess_frame, validate_frame
from train import train
from evaluate import evaluate


def show_menu():
    print("\n" + "=" * 50)
    print("Real-Time Surgical Tool Segmentation System")
    print("=" * 50)
    print("1. Train Model")
    print("2. Evaluate Model")
    print("3. Run Real-Time Simulation")
    print("4. Show Model Info")
    print("5. Change Confidence Threshold")
    print("0. Exit")


def run_realtime(model):
    print("\nStarting real-time simulation...\n")

    frames = [
        "live_frame_001",
        "live_frame_002",
        "live_frame_003",
        "live_frame_004"
    ]

    for frame_name in frames:
        print("\nProcessing:", frame_name)

        frame = preprocess_frame(frame_name)

        if not validate_frame(frame):
            continue

        predictions = model.predict(frame)
        filtered = model.filter_predictions(predictions)

        if not filtered:
            print("No tools detected.")
        else:
            for p in filtered:
                print(
                    p["label"],
                    "| confidence:", p["confidence"],
                    "| mask:", p["mask"]
                )


def change_threshold(model):
    value = input("Enter new threshold (0-1): ")
    try:
        model.set_confidence_threshold(float(value))
    except:
        print("Invalid value")


def main():
    model = SegmentationModel()
    model.load_weights()

    while True:
        show_menu()
        choice = input("Enter choice: ")

        if choice == "1":
            train()

        elif choice == "2":
            evaluate()

        elif choice == "3":
            run_realtime(model)

        elif choice == "4":
            model.show_model_info()

        elif choice == "5":
            change_threshold(model)

        elif choice == "0":
            print("Exiting...")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
