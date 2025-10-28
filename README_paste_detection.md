# Paste Detection API

A FastAPI service for detecting and calculating paste areas using YOLO detection + Mobile SAM segmentation.

## Architecture

**YOLO Detection → Mobile SAM Segmentation**

1. YOLO model detects paste objects and returns the bounding box closest to image center
2. Mobile SAM performs precise pixel-level segmentation on the detected area
3. Returns total inference time and paste pixel count for each image

## Files

- `paste_detection_server.py` - FastAPI server for paste detection inference
- `paste_inference.py` - Core inference engine with YOLO + Mobile SAM
- `trigger_paste_detection.py` - Client to trigger inference requests
- `requirements_paste_detection.txt` - Python dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements_paste_detection.txt
```

2. Ensure YOLO model exists at:
```
models/yolo_detection/paste.pt
```

3. Mobile SAM model will be downloaded automatically on first run.

## Usage

### Start the Server

```bash
python paste_detection_server.py --port 8001
```

Options:
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8001)
- `--yolo-model`: Path to YOLO paste model (default: models/yolo_detection/paste.pt)

### Trigger Inference

```bash
python trigger_paste_detection.py /path/to/job/folder
```

Options:
- `--server-url`: Server URL (default: http://127.0.0.1:8001)
- `--extensions`: Image extensions to process (e.g., .jpg .png)
- `--wait`: Wait for server to be ready
- `--output-file`: Custom output CSV file path
- `--no-save`: Don't save results to CSV

### API Endpoints

- `GET /health` - Check server health and model readiness
- `POST /inference` - Run paste detection on job folder
- `GET /docs` - Interactive API documentation

## API Response Format

```json
{
  "status": "success",
  "message": "Successfully processed N images",
  "total_images": 10,
  "total_inference_time": 25.5,
  "results": [
    {
      "image_path": "/path/to/image.jpg",
      "paste_pixels": 1250,
      "bbox": [100, 150, 300, 350],
      "inference_time": 2.1
    }
  ]
}
```

## Integration with Existing Infrastructure

The paste detection API follows the same pattern as `patchcore_server.py` and can be called using the existing `trigger_inference.py` framework by updating the server URL to point to the paste detection service.