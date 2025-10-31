# Automatic License Plate Recognition

The application detects and tracks vehicles in video footage, identifies license plates, and uses OCR to read the plate text. It is designed for Swedish license plates, validating them against standard formats such as LLLNNL or LLLNNN (L = letter, N = number).

## Models
- **YOLOv11m** for vehicle detection.
- **Fine-tuned YOLOv11n** trained on a custom license plate dataset for license plate detection.
- **PaddleOCR** for text recognition on detected plates.
- **OpenCV** for video handling and annotations.

## Installation
- Python 3.13+
- Install dependencies with uv:
    ```bash
    # For cuda
    uv sync --group cuda
    ```
    ```bash
    # For cpu
    uv sync --group cpu
    ```


## Usage
Run the main script with a video file as input:
```
uv run app/main.py path/to/video.mp4 [options]
```

### Command-Line Arguments
  - `VIDEO_SOURCE`: Path to the video file. Required.
  - `--save-video`: Save annotated video output (default: False).
  - `--vehicle-conf`: Confidence threshold for vehicle detection (default: 0.6).
  - `--plate-conf`: Confidence threshold for plate detection (default: 0.7).
  - `--ocr-conf`: Confidence threshold for OCR (default: 0.85).
  - `--ocr-conf-retry-threshold`: Max confidence to retry OCR (default: 0.95).
 
Type --help for more information.

### Example
Process a video and save annotated output:
```
uv run app/main.py input.mp4 --save-video
```

## Output
Results are saved in a timestamped directory in `results/` directory:
- `results.csv`: CSV with columns: Timestamp (s), Plate Text, Confidence, Car Image, Plate Image.
- Cropped vehicle and plate images.
- `config.json`: Dump of the used configuration.
- `output.mp4`: Annotated video (if `--save-video` is used).

## Configuration
All defaults are defined in `config.py`. You can override them via command-line arguments or modify the `Config` class directly.
- **Logging:** Set environment variable `LOGLEVEL` to `DEBUG`, `INFO`, etc. (default: INFO).
- **Debugging:** Enable DEBUG logging to save each prediction attempt in `debug/`.

## License
MIT License
