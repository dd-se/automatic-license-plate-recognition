import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from config import CONFIG
from logging_helper import get_logger
from paddleocr import PaddleOCR
from ultralytics.models import YOLO

logger = get_logger(__name__)


@dataclass
class OcrResult:
    input: str | list[str]
    output: str
    ocr_conf: float
    valid: bool
    reason: str
    index: int | None = None
    plate_crop: np.ndarray | None = None

    def __bool__(self):
        return self.valid

    def __str__(self):
        return f"OcrResult(input='{self.input}', output='{self.output}', conf={self.ocr_conf}, valid={self.valid}, reason='{self.reason}', index={self.index})"


class PlateValidator:
    KEEP_ALPHANUMERIC = re.compile(r"[^A-Z0-9]")
    SWEDISH_PLATE_PATTERN = re.compile(r"^[A-HJ-NOPR-TUWXYZ]{3}\d{2}(?:\d|[A-HJ-NPR-TUWXYZ])$")

    @classmethod
    def clean_text(cls, texts: list[str]) -> str:
        cleaned = re.sub(cls.KEEP_ALPHANUMERIC, "", "".join(texts))

        if cleaned.endswith("O"):
            cleaned = cleaned[:-1] + "0"
        return cleaned

    @classmethod
    def validate(cls, texts: list[str] | str, ocr_conf: list[float]) -> OcrResult:
        cleaned = cls.clean_text(texts)
        ocr_conf = round(np.mean(ocr_conf), 3)
        if len(cleaned) != 6:
            return OcrResult(
                texts,
                cleaned,
                ocr_conf,
                False,
                f"Invalid length ({len(cleaned)} chars, expected 6)",
            )

        if not cls.SWEDISH_PLATE_PATTERN.match(cleaned):
            return OcrResult(
                texts,
                cleaned,
                ocr_conf,
                False,
                "Format does not match Swedish plate pattern (LLLNNL or LLLNNN)",
            )

        return OcrResult(texts, cleaned, ocr_conf, True, "Valid Swedish license plate")


class PlateProcessor:
    """Handles detection and OCR of license plates."""

    def __init__(self, plate_detector: YOLO, paddle_ocr: PaddleOCR):
        self.debug_counter = 0
        self.debug_dir = CONFIG.DEBUG_DIR / f"{CONFIG.VIDEO_SOURCE.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.plate_detector = plate_detector
        self.paddle_ocr = paddle_ocr

    def _preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        processed = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

    def _read_plates(self, vehicle_crop: np.ndarray, plate_crops: list[np.ndarray]) -> OcrResult | None:
        best_score = 0
        ocr_result: OcrResult | None = None
        results = self.paddle_ocr.predict(plate_crops)
        for idx, result in enumerate(results):
            texts = result.get("rec_texts")
            scores = result.get("rec_scores")
            if texts and scores:
                valid_ocr_result = PlateValidator.validate(texts, scores)
                valid_ocr_result.index = idx
                valid_ocr_result.plate_crop = plate_crops[idx]

                if valid_ocr_result and valid_ocr_result.ocr_conf > best_score:
                    best_score = valid_ocr_result.ocr_conf
                    ocr_result = valid_ocr_result

                if logger.isEnabledFor(logging.DEBUG):
                    self.debug_save(vehicle_crop, valid_ocr_result)

        return ocr_result

    def process(self, vehicle_crop: np.ndarray) -> OcrResult | None:
        boxes = self.plate_detector.predict(vehicle_crop, conf=CONFIG.PLATE_CONF)[0].boxes

        if boxes:
            bounding_boxes = boxes.xyxy.int()
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = bbox

                plate_crop = vehicle_crop[y1:y2, x1:x2]
                plates = [
                    plate_crop,
                    cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                ]
                ocr_result = self._read_plates(vehicle_crop, plates)
                return ocr_result

        return None

    def debug_save(self, vehicle_crop: np.ndarray, ocr_result: OcrResult):
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_counter += 1
        plate_text = ocr_result.output
        idx = ocr_result.index
        debug_conf = int(ocr_result.ocr_conf * 100) if ocr_result.ocr_conf else "00"
        vehicle_crop_path = self.debug_dir / f"{self.debug_counter}-{plate_text}_vehicle_idx{idx}_conf{debug_conf}.png"
        plate_crop_path = self.debug_dir / f"{self.debug_counter}-{plate_text}_plate_idx{idx}_conf{debug_conf}.png"
        cv2.imwrite(vehicle_crop_path, vehicle_crop)
        cv2.imwrite(plate_crop_path, ocr_result.plate_crop)

        logger.debug(ocr_result)
