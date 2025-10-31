import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from config import CONFIG
from logging_helper import get_logger

logger = get_logger(__name__)


class ResultsGenerator:
    def __init__(self):
        self.counter = 0
        self.predictions: dict[int, Any] = {}
        self.results_dir = CONFIG.OUTPUT_DIR / f"{CONFIG.VIDEO_SOURCE.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file_path = self.results_dir / "results.csv"
        self.config_path = self.results_dir / "config.json"
        logger.info(f"Results will be saved in: {self.results_dir}")

    def save_prediction(
        self,
        track_id: int,
        timestamp: float,
        plate_text: str,
        confidence: float,
        vehicle_crop: np.ndarray,
        plate_crop: np.ndarray,
        index: int,
    ):
        self.counter += 1
        timestamp_int = int(timestamp)
        car_img_path = self.results_dir / f"{self.counter}-{plate_text}_vehicle_{timestamp_int}s_idx{index}.png"
        plate_img_path = self.results_dir / f"{self.counter}-{plate_text}_plate_{timestamp_int}s_idx{index}.png"
        self.predictions[track_id] = (
            timestamp,
            plate_text,
            confidence,
            car_img_path,
            plate_img_path,
            vehicle_crop,
            plate_crop,
        )

    def write_results(self):
        with open(self.csv_file_path, "w", encoding="utf-8") as f:
            f.write("Timestamp (s),Plate Text,Confidence,Car Image,Plate Image\n")
            for (
                timestamp,
                plate_text,
                confidence,
                car_img_path,
                plate_img_path,
                vehicle_crop,
                plate_crop,
            ) in self.predictions.values():
                cv2.imwrite(f"{car_img_path}", vehicle_crop)
                cv2.imwrite(f"{plate_img_path}", plate_crop)
                f.write(f"{timestamp:.2f},{plate_text},{confidence},{car_img_path.name},{plate_img_path.name}\n")

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(CONFIG.__dict__, f, default=str, indent=2)
