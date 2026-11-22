"""Shared code for the whole project: class list, settings and preprocessing.

EVERYTHING that touches an image (training notebooks on Colab AND the local
webcam app) must go through the functions in this file, so that the models
see exactly the same kind of input during training and during the live demo.

Only change settings here after telling the group, because changing them
means both models have to be retrained.
"""

from pathlib import Path
import os
import random

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Classes: the single source of truth for label <-> index.
# DEL and SPACE from the dataset are deliberately excluded.
# ---------------------------------------------------------------------------
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]  # 'A'..'Z'
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)  # 26

# ---------------------------------------------------------------------------
# Paths. Override the data location with the ASL_DATA_DIR environment variable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ASL_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw" / "asl_raw"      # Mendeley Type_01 raw photos
SUBSET_DIR = DATA_DIR / "subset"
SPLITS_DIR = DATA_DIR / "splits"
WEBCAM_TRAIN_DIR = DATA_DIR / "webcam_train"
WEBCAM_TEST_DIR = DATA_DIR / "webcam_test"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SEED = 467
IMAGES_PER_CLASS = 500                    # subset size per letter (proposal)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)      # train / val / test

# HOG + SVM input
HOG_IMG_SIZE = 64                         # grayscale square fed to HOG
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

# ResNet18 input (ImageNet statistics, because the backbone is ImageNet-pretrained)
CNN_IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Webcam: fixed square box the user puts their hand in.
# Given as fractions of the frame so it works at any camera resolution.
# Dataset images are tightly cropped around the hand, so the hand should fill the box.
BOX_CENTER = (0.70, 0.50)                 # (x, y) as fraction of width / height
BOX_SIZE = 0.55                           # side length as fraction of frame height
MIRROR_WEBCAM = True                      # flip preview like a mirror (selfie view)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Make random choices reproducible (numpy, python, and torch if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_image(path) -> np.ndarray:
    """Read an image from disk as a BGR uint8 array (OpenCV's default order)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def box_coords(frame_shape) -> tuple[int, int, int, int]:
    """Pixel coordinates (x1, y1, x2, y2) of the fixed hand box for a frame."""
    h, w = frame_shape[:2]
    side = int(BOX_SIZE * h)
    cx, cy = int(BOX_CENTER[0] * w), int(BOX_CENTER[1] * h)
    x1 = max(0, min(w - side, cx - side // 2))
    y1 = max(0, min(h - side, cy - side // 2))
    return x1, y1, x1 + side, y1 + side


def crop_box(frame: np.ndarray) -> np.ndarray:
    """Cut the fixed hand box out of a webcam frame (BGR in, BGR out)."""
    x1, y1, x2, y2 = box_coords(frame.shape)
    return frame[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------------
# Preprocessing: used for dataset images AND webcam crops
# ---------------------------------------------------------------------------
def preprocess_hog(img_bgr: np.ndarray) -> np.ndarray:
    """BGR image -> HOG_IMG_SIZE x HOG_IMG_SIZE grayscale float32 in [0, 1]."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (HOG_IMG_SIZE, HOG_IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def hog_features(img_bgr: np.ndarray) -> np.ndarray:
    """BGR image -> 1-D HOG feature vector (input to the SVM)."""
    from skimage.feature import hog  # imported here so the CNN side doesn't need it
    return hog(preprocess_hog(img_bgr), feature_vector=True, **HOG_PARAMS).astype(np.float32)


def prepare_rgb(img_bgr: np.ndarray, size: int = CNN_IMG_SIZE) -> np.ndarray:
    """BGR image -> RGB uint8 resized to size x size (before augmentation/normalization)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def normalize_cnn(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 3xHxW normalized with ImageNet mean/std."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)


def preprocess_cnn(img_bgr: np.ndarray) -> np.ndarray:
    """BGR image -> float32 3x224x224, ready for ResNet18 (wrap with torch.from_numpy)."""
    return normalize_cnn(prepare_rgb(img_bgr))
