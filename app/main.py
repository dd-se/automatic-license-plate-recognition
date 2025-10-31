import os
from dataclasses import dataclass
from queue import Empty, Queue

from config import CONFIG

os.environ["YOLO_VERBOSE"] = "False"
import threading

import cv2
import numpy as np
import paddle
import torch
from logging_helper import get_logger
from paddleocr import PaddleOCR
from plate_processor import PlateProcessor
from results_generator import ResultsGenerator
from ultralytics.models import YOLO

logger = get_logger(__name__)


@dataclass
class Vehicle:
    bounding_box: list[int]
    last_update_frame: int
    frame_time_seconds: float = 0.0
    vehicle_crop: np.ndarray | None = None
    plate_text: str | None = None
    ocr_conf: float = 0
    index: int = -1


class ModelLoader:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Paddle using device: {paddle.get_device()}")
        logger.debug(f"YOLO using device: {device}")
        self.vehicle_detector = self._load_yolo(CONFIG.VEHICLE_MODEL_PATH, device)
        self.plate_detector = self._load_yolo(CONFIG.PLATE_MODEL_PATH, device)
        self.ocr_model = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=CONFIG.OCR_CONF,
        )

    def _load_yolo(self, model_path: str, device: str) -> YOLO:
        model = YOLO(model_path).to(device)
        return model


def annotate_in_background(
    VEHICLES_TRACKED: dict[int, Vehicle],
    queue: Queue,
    video_out: cv2.VideoWriter,
    width: int,
    height: int,
    resize_frame: bool,
    stop: threading.Event,
):
    while True:
        try:
            frame_counter, frame = queue.get(timeout=1)

            for track_id, vehicle in VEHICLES_TRACKED.copy().items():
                # Delete stale tracks
                frames_since_seen = frame_counter - vehicle.last_update_frame
                if frames_since_seen > CONFIG.STALE_TRACK_ID_THRESHOLD:
                    logger.debug(f"Stale Track ID removed: {track_id}")
                    del VEHICLES_TRACKED[track_id]
                    continue

                # Draw bounding box and label
                x1, y1, x2, y2 = vehicle.bounding_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), CONFIG.BOX_COLOR, CONFIG.BOX_THICKNESS)
                text = vehicle.plate_text or f"ID: {track_id}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text,
                    CONFIG.TEXT_FONT,
                    CONFIG.TEXT_FONT_SCALE,
                    CONFIG.TEXT_THICKNESS,
                )
                cv2.rectangle(
                    frame,
                    (x1 - 1, y1 - text_h - baseline - 10),
                    (x1 + text_w + 8, y1),
                    CONFIG.TEXT_BG_COLOR,
                    -1,
                )
                cv2.putText(
                    frame,
                    text,
                    (x1 + 4, y1 - 12),
                    CONFIG.TEXT_FONT,
                    CONFIG.TEXT_FONT_SCALE,
                    CONFIG.TEXT_COLOR,
                    CONFIG.TEXT_THICKNESS,
                )

            if resize_frame:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            video_out.write(frame)

        except Empty:
            if stop.is_set():
                video_out.release()
                break
        except Exception as e:
            logger.error(e, exc_info=True)
            break


def main():
    VEHICLES_TRACKED: dict[int, Vehicle] = {}
    models = ModelLoader()
    plate_processor = PlateProcessor(models.plate_detector, models.ocr_model)
    results_generator = ResultsGenerator()

    cap = cv2.VideoCapture(CONFIG.VIDEO_SOURCE)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info(f"Video properties: {frame_width:}x{frame_height} at {fps} FPS")

    if CONFIG.VIDEO_OUT:
        # If resolution is greater than 1080p, halve the dimensions
        resize_frame = False
        if frame_width > 1920 or frame_height > 1080:
            frame_width //= 2
            frame_height //= 2
            resize_frame = True

        video_out = cv2.VideoWriter(
            results_generator.results_dir / "output.mp4",
            cv2.VideoWriter.fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        annotation_queue = Queue()
        stop_annotation_thread = threading.Event()
        annotation_thread = threading.Thread(
            target=annotate_in_background,
            args=(
                VEHICLES_TRACKED,
                annotation_queue,
                video_out,
                frame_width,
                frame_height,
                resize_frame,
                stop_annotation_thread,
            ),
            daemon=True,
        )
        annotation_thread.start()

    frame_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % CONFIG.FRAME_SKIP_RATE != 0:
                continue

            boxes = models.vehicle_detector.track(
                frame,
                persist=True,
                tracker=CONFIG.TRACKER_CONFIG,
                classes=CONFIG.VEHICLE_CLASSES,
                conf=CONFIG.VEHICLE_CONF,
            )[0].boxes

            if boxes and boxes.is_track:
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                bounding_boxes = boxes.xyxy.int().numpy()
                track_ids = boxes.id.int().numpy()

                for track_id, bbox in zip(track_ids, bounding_boxes):
                    x1, y1, x2, y2 = bbox
                    vehicle_crop = frame[y1:y2, x1:x2]

                    if track_id not in VEHICLES_TRACKED:
                        VEHICLES_TRACKED[track_id] = Vehicle(
                            bounding_box=bbox,
                            frame_time_seconds=timestamp_sec,
                            vehicle_crop=vehicle_crop,
                            last_update_frame=frame_counter,
                        )
                    elif VEHICLES_TRACKED[track_id].ocr_conf >= CONFIG.OCR_CONF_RETRY_THRESHOLD:
                        # Update vehicle position in frame and continue to next vehicle when confident
                        # Set OCR_CONF_RETRY_THRESHOLD to the same value as OCR_CONF to skip retries.
                        VEHICLES_TRACKED[track_id].bounding_box = bbox
                        VEHICLES_TRACKED[track_id].last_update_frame = frame_counter
                        continue
                    else:
                        # Update existing vehicle info
                        VEHICLES_TRACKED[track_id].bounding_box = bbox
                        VEHICLES_TRACKED[track_id].vehicle_crop = vehicle_crop
                        VEHICLES_TRACKED[track_id].frame_time_seconds = timestamp_sec
                        VEHICLES_TRACKED[track_id].last_update_frame = frame_counter

                    vehicle = VEHICLES_TRACKED[track_id]
                    ocr_result = plate_processor.process(vehicle.vehicle_crop)
                    if ocr_result and ocr_result.ocr_conf > vehicle.ocr_conf:
                        if vehicle.ocr_conf > 0:
                            logger.info(
                                f"Track ID {track_id}: [+] {ocr_result.output} | Conf: {ocr_result.ocr_conf * 100:.1f}% | Idx: {ocr_result.index} -> (Prev: {vehicle.plate_text}, {vehicle.ocr_conf * 100:.1f}%, {vehicle.index})"
                            )
                        else:
                            logger.info(
                                f"Track ID {track_id}: {ocr_result.output} | Conf: {ocr_result.ocr_conf * 100:.1f}% | Idx: {ocr_result.index}"
                            )

                        vehicle.plate_text = ocr_result.output
                        vehicle.ocr_conf = ocr_result.ocr_conf
                        vehicle.index = ocr_result.index

                        results_generator.save_prediction(
                            track_id,
                            vehicle.frame_time_seconds,
                            vehicle.plate_text,
                            vehicle.ocr_conf,
                            vehicle.vehicle_crop,
                            ocr_result.plate_crop,
                            vehicle.index,
                        )

            if CONFIG.VIDEO_OUT:
                annotation_queue.put((frame_counter, frame.copy()))

    finally:
        if CONFIG.VIDEO_OUT:
            stop_annotation_thread.set()
            annotation_thread.join()
        cap.release()
        results_generator.write_results()
        logger.info("Processing finished")


if __name__ == "__main__":
    main()
