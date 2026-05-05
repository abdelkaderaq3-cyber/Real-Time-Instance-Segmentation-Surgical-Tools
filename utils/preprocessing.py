import numpy as np


def preprocess_frame(frame):
    print("Preprocessing frame...")

    frame = to_array(frame)
    frame = normalize(frame)
    frame = resize(frame)

    return frame


def to_array(frame):
    return np.ones((100, 100))


def normalize(frame):
    return frame / 255.0


def resize(frame):
    return np.zeros((224, 224))


def augment_frame(frame):
    return {
        "original": frame,
        "flip": frame,
        "rotate": frame
    }


def validate_frame(frame):
    if frame is None:
        print("Invalid frame")
        return False

    return True
