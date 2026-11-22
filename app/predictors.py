"""Load the two trained models and run them on one cropped hand image.

Shared by webcam_app.py (live demo) and benchmark.py (CPU timing), so both use
exactly the same code path. Preprocessing goes through common.py (project rule 1).

    from predictors import load_available
    models = load_available()                    # {"hog": HogSvm, "cnn": ResNet18}
    pred = models["cnn"].predict(crop_bgr)       # Prediction(letter, confidence, top3, ...)
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

HOG_PATH = common.MODELS_DIR / "hog_svm.joblib"
CNN_PATH = common.MODELS_DIR / "resnet18.pt"


@dataclass
class Prediction:
    letter: str
    confidence: float                  # probability of `letter` (score if the model is uncalibrated)
    probs: np.ndarray                  # one value per class, in self.classes order
    top3: list                         # [(letter, value), ...]
    preprocess_ms: float
    inference_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Predictor:
    """Base class: subclasses implement prepare() (preprocessing) and scores() (the model)."""
    key = ""
    name = ""
    classes: list
    calibrated = True                  # False -> confidence is a softmax of SVM margins, not a probability

    def prepare(self, crop_bgr: np.ndarray):
        raise NotImplementedError

    def scores(self, x) -> np.ndarray:
        raise NotImplementedError

    def predict(self, crop_bgr: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        x = self.prepare(crop_bgr)
        t1 = time.perf_counter()
        p = self.scores(x)
        t2 = time.perf_counter()
        return self.make_prediction(p, (t1 - t0) * 1e3, (t2 - t1) * 1e3)

    def make_prediction(self, p: np.ndarray, pre_ms: float = 0.0, inf_ms: float = 0.0) -> Prediction:
        order = np.argsort(-p)
        return Prediction(letter=self.classes[order[0]], confidence=float(p[order[0]]), probs=p,
                          top3=[(self.classes[i], float(p[i])) for i in order[:3]],
                          preprocess_ms=pre_ms, inference_ms=inf_ms)

    def warmup(self, n: int = 3) -> None:
        dummy = np.zeros((300, 300, 3), np.uint8)
        for _ in range(n):
            self.predict(dummy)


class HogSvm(Predictor):
    """HOG features (common.hog_features) + linear SVM, from models/hog_svm.joblib.

    Accepts either a bare scikit-learn estimator/Pipeline, or a dict holding one under
    "model" / "pipeline" / "clf" / "svm" (optionally with "classes"). The estimator's
    classes_ may be letters ('A'..'Z') or indices (0..25).
    """
    key = "hog"
    name = "HOG + SVM"

    def __init__(self, path=HOG_PATH):
        import joblib
        obj = joblib.load(path)
        extra_classes = None
        if isinstance(obj, dict):
            found = [k for k in ("model", "pipeline", "clf", "svm") if k in obj]
            if not found:
                raise ValueError(f"{path}: dict has no 'model' entry (keys: {list(obj)})")
            extra_classes = obj.get("classes")
            obj = obj[found[0]]
        self.model = obj
        cols = list(self.model.classes_)          # column order of decision_function / predict_proba
        lookup = list(extra_classes) if extra_classes is not None else common.CLASSES
        self.classes = [str(c) if isinstance(c, str) else str(lookup[int(c)]) for c in cols]
        self.calibrated = hasattr(self.model, "predict_proba")

    def prepare(self, crop_bgr):
        return common.hog_features(crop_bgr)[None]

    def scores(self, x):
        if self.calibrated:
            return self.model.predict_proba(x)[0]
        return softmax(self.model.decision_function(x)[0])


class ResNet18(Predictor):
    """Fine-tuned ResNet18 from models/resnet18.pt (export format of notebooks/3_resnet18.ipynb)."""
    key = "cnn"
    name = "ResNet18"

    def __init__(self, path=CNN_PATH, threads: int | None = None):
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        ck = torch.load(path, map_location="cpu")
        self.classes = list(ck["classes"])
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(ck["state_dict"])
        self.model = model.eval()
        self.info = {k: v for k, v in ck.items() if k != "state_dict"}

    def prepare(self, crop_bgr):
        return self.torch.from_numpy(common.preprocess_cnn(crop_bgr))[None]

    def scores(self, x):
        with self.torch.inference_mode():
            return self.torch.softmax(self.model(x), 1)[0].numpy()


def load_available(which=("hog", "cnn"), threads: int | None = None, verbose: bool = True) -> dict:
    """Load whichever model files exist. Returns {key: Predictor}; missing ones are skipped."""
    loaders = {"hog": (HOG_PATH, lambda: HogSvm()), "cnn": (CNN_PATH, lambda: ResNet18(threads=threads))}
    models = {}
    for key in which:
        path, load = loaders[key]
        if not path.exists():
            if verbose:
                print(f"[skip] {path.relative_to(common.PROJECT_ROOT)} not found (download it from the shared Drive into models/)")
            continue
        models[key] = load()
        models[key].warmup()
        if verbose:
            note = "" if models[key].calibrated else "  (uncalibrated: shows SVM scores, not probabilities)"
            print(f"[ok]   {models[key].name} loaded from {path.relative_to(common.PROJECT_ROOT)}{note}")
    return models
