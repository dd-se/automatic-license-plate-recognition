import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2


@dataclass
class Config:
    ROOT_DIR: Path = Path(__file__).parent.parent
    MODEL_DIR: Path = ROOT_DIR / "models"

    VIDEO_SOURCE: Path = None
    VIDEO_OUT: bool = False
    FRAME_SKIP_RATE: int = 1
    OUTPUT_DIR: Path = ROOT_DIR / "results"

    VEHICLE_MODEL_PATH: Path = MODEL_DIR / "yolo11m.pt"
    VEHICLE_CLASSES: list[int] = field(default_factory=lambda: [2, 3, 5, 7])  # car, motorcycle, bus, truck
    TRACKER_CONFIG: Path = MODEL_DIR / "bytetrack.yaml"
    VEHICLE_CONF: float = 0.6
    STALE_TRACK_ID_THRESHOLD: int = 45

    PLATE_MODEL_PATH: Path = MODEL_DIR / "yolo11n_plate_detector.pt"
    PLATE_CONF: float = 0.7
    OCR_CONF: float = 0.85
    OCR_CONF_RETRY_THRESHOLD: float = 0.95

    BOX_COLOR: tuple = (254, 101, 216)
    BOX_THICKNESS: int = 2
    TEXT_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_FONT_SCALE: float = 1.2
    TEXT_THICKNESS: int = 2
    TEXT_BG_COLOR: tuple = (254, 101, 216)
    TEXT_COLOR: tuple = (0, 0, 0)

    DEBUG_DIR: Path = ROOT_DIR / "debug"


CONFIG = Config()


parser = argparse.ArgumentParser(description="Vehicle and License Plate Recognition Configuration")

# Video & output
parser.add_argument("VIDEO_SOURCE", type=Path, help="Path to video file")
parser.add_argument(
    "--save-video",
    dest="VIDEO_OUT",
    action="store_true",
    default=CONFIG.VIDEO_OUT,
    help="Save the annotated frames as a video file",
)

parser.add_argument(
    "--frame-skip-rate",
    dest="FRAME_SKIP_RATE",
    type=int,
    default=CONFIG.FRAME_SKIP_RATE,
    help="Number of frames to skip between processing",
)
parser.add_argument(
    "--output-dir", dest="OUTPUT_DIR", type=Path, default=CONFIG.OUTPUT_DIR, help="Directory for saving results"
)

# Vehicle detection
parser.add_argument(
    "--vehicle-model-path",
    dest="VEHICLE_MODEL_PATH",
    type=Path,
    default=CONFIG.VEHICLE_MODEL_PATH,
    help="Path to YOLO vehicle detection model",
)
parser.add_argument(
    "--vehicle-classes",
    dest="VEHICLE_CLASSES",
    type=int,
    nargs="+",
    default=CONFIG.VEHICLE_CLASSES,
    help="List of vehicle class IDs to track",
)
parser.add_argument(
    "--tracker-config",
    dest="TRACKER_CONFIG",
    type=Path,
    default=CONFIG.TRACKER_CONFIG,
    help="Path to tracker configuration YAML",
)
parser.add_argument(
    "--vehicle-conf",
    dest="VEHICLE_CONF",
    type=float,
    default=CONFIG.VEHICLE_CONF,
    help="Confidence threshold for vehicle detection",
)
parser.add_argument(
    "--stale-track-id-threshold",
    dest="STALE_TRACK_ID_THRESHOLD",
    type=int,
    default=CONFIG.STALE_TRACK_ID_THRESHOLD,
    help="Number of frames to keep a track without updates",
)

# Plate & OCR
parser.add_argument(
    "--plate-model-path",
    dest="PLATE_MODEL_PATH",
    type=Path,
    default=CONFIG.PLATE_MODEL_PATH,
    help="Path to license plate detection model",
)
parser.add_argument(
    "--plate-conf",
    dest="PLATE_CONF",
    type=float,
    default=CONFIG.PLATE_CONF,
    help="Confidence threshold for plate detection",
)
parser.add_argument(
    "--ocr-conf", dest="OCR_CONF", type=float, default=CONFIG.OCR_CONF, help="Confidence threshold for OCR results"
)
parser.add_argument(
    "--ocr-conf-retry-threshold",
    dest="OCR_CONF_RETRY_THRESHOLD",
    type=float,
    default=CONFIG.OCR_CONF_RETRY_THRESHOLD,
    help="Max confidence threshold for retrying OCR",
)

args = parser.parse_args()
for key, value in args.__dict__.items():
    setattr(CONFIG, key, value)
if not CONFIG.VIDEO_SOURCE.is_file():
    raise FileNotFoundError(f"Error: The file '{CONFIG.VIDEO_SOURCE}' does not exist.")
