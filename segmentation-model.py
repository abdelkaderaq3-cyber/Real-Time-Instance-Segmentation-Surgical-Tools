import random


class SegmentationModel:
    def __init__(self):
        self.classes = ["scissors", "forceps", "scalpel", "needle"]
        self.threshold = 0.5

    def load_weights(self):
        print("Model weights loaded")

    def predict(self, frame):
        results = []

        for i, tool in enumerate(self.classes):
            conf = round(0.6 + i * 0.1 + random.uniform(-0.05, 0.05), 2)

            results.append({
                "label": tool,
                "confidence": conf,
                "mask": f"{tool}_mask"
            })

        return results

    def filter_predictions(self, preds):
        return [p for p in preds if p["confidence"] >= self.threshold]

    def set_confidence_threshold(self, value):
        self.threshold = value
        print("Threshold updated to", value)

    def show_model_info(self):
        print("Classes:", self.classes)
        print("Threshold:", self.threshold)
