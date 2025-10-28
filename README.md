# AI Inference API - PatchCore Anomaly Detection

This project implements a FastAPI-based inference service for anomaly detection using PatchCore models. The system is designed to process industrial inspection images and detect anomalies in manufacturing processes.

## Project Overview

The AI inference system processes job folders containing images through multiple AI models to detect various types of defects and anomalies. The current implementation focuses on PatchCore anomaly detection with FastAPI serving capabilities.

## Architecture & Inference Process

### Original Complete AI Pipeline (Reference: ai_workflow_diagram.txt)

The complete AI inference workflow involves multiple models working in parallel and sequential processing:

```
Job Detection → Image Loading → AI Inference → Results Integration → Output Generation
```

#### Phase 1: Job Detection
- **Input**: Job folder (`W:\Machine\{timestamp}`)
- **Trigger**: 02.exe Go application detects ready job folders
- **Validation**: Checks for corresponding CSV file (`AMR\{timestamp}.csv`)

#### Phase 2: Image Loading
- **Process**: Loads images in batches of 4 from CSV paths
- **Preprocessing**: Prepares images for model inference

#### Phase 3: AI Inference (Multi-Model Processing)

**Parallel Processing (Batch Size = 4):**
1. **Anomaly Detection** (PatchCore + EfficientNet-B5)
   - Processing time: ~0.8s for batch of 4 images
   - Threshold: 0.46
   - Output: `abnormal_scores` + `ai_defect_names`

2. **Pad Detection** (YOLO Pad Model)
   - Processing time: ~0.3s for batch of 4 images
   - Output: Pad centers + minimum inter-pad distances

**Sequential Processing:**
3. **Paste Detection** (YOLO Paste Model → SAM Segmentation)
   - YOLO: ~0.3s for batch of 4 images (detects paste bboxes)
   - SAM: ~1.2s per bbox (BOTTLENECK - sequential processing)
   - Output: Pixel counts (`insp_area_ai`)

#### Phase 4: Results Integration
- **Defect Classification** (Priority-based):
  1. Volume defects (inspection volume thresholds)
  2. Height defects (inspection height > 200)
  3. AI anomaly defects (abnormal_score > 0.46)
  4. Distance defects (min_pad_distance < 6.6 AND coverage > 180%)
  5. Coverage defects (coverage > 180%)
- **Final Judgment**: `ai_juge = "NG"` if defects detected, `"OK"` otherwise

#### Phase 5: Output Generation
- **Primary Output**: `W:\Machine\{timestamp}\AI\{timestamp}.csv`
- **Backup Output**: `inference_result_path\YYYYMMDD\`

### Performance Metrics (Complete Pipeline)
- **Total Processing Time**: ~2.8s + (1.2s × total_detected_paste_bboxes)
- **Example**: 4 images with 1 bbox each = ~7.6s total
- **GPU Memory**: ~10GB VRAM total
- **Bottleneck**: SAM segmentation (sequential processing)

## Current Implementation Status

### ✅ Implemented Components

#### 1. PatchCore Inference Engine (`patchcore_inference.py`)
- **Model Format**: ONNX for optimized inference
- **GPU Support**: DirectML (Windows) with CPU fallback
- **Input Processing**: 224x224 image preprocessing
- **Batch Processing**: Supports multiple image processing
- **Output**: Anomaly scores and binary classification

#### 2. FastAPI Server (`patchcore_server.py`)
- **Framework**: FastAPI with async support
- **Endpoints**:
  - `GET /health` - Server health check
  - `POST /inference` - Trigger inference on job folder
  - `GET /` - API information
  - `GET /docs` - Auto-generated API documentation
- **Features**:
  - Automatic model initialization on startup
  - Job folder validation
  - Batch image processing
  - Statistical result aggregation
  - Error handling with HTTP status codes

#### 3. Client Trigger (`trigger_inference.py`)
- **HTTP Client**: Requests-based client for server communication
- **Features**:
  - Server health monitoring
  - Inference triggering with timeout handling
  - Result saving to CSV
  - Command-line interface

### 📋 Current Workflow

```
Client Request → FastAPI Server → PatchCore ONNX Model → Results Processing → JSON Response
```

1. **Server Startup**: PatchCore model loaded into memory
2. **Client Request**: Job folder path sent via POST `/inference`
3. **Image Processing**: Batch processing of images in job folder
4. **Inference**: ONNX model processes preprocessed images
5. **Response**: JSON with anomaly scores, statistics, and detailed results

## Installation & Setup

### Prerequisites
- Python 3.8+
- ONNX Runtime
- OpenCV
- FastAPI
- Uvicorn

### Installation
```bash
pip install fastapi uvicorn onnxruntime opencv-python pandas requests
```

### Model Setup
1. Place PatchCore ONNX model in `models/patchcore/` directory
2. Ensure model file has `.onnx` extension

## Usage

### 1. Start the Server
```bash
python patchcore_server.py --host 0.0.0.0 --port 8000 --model-folder models/patchcore
```

### 2. Check Server Health
```bash
curl http://localhost:8000/health
```

### 3. Trigger Inference (Command Line)
```bash
python trigger_inference.py /path/to/job/folder --wait --extensions .jpg .png
```

### 4. Trigger Inference (API)

On Linux/macOS or Git Bash:
```bash
curl -X POST "http://localhost:8000/inference" \
     -H "Content-Type: application/json" \
     -d '{"job_folder": "/path/to/job/folder"}'
```

On Windows PowerShell, either use PowerShell-native or the real curl binary:

PowerShell-native (recommended):
```powershell
$body = @{ job_folder = 'D:/path/to/job/folder' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8000/inference' -Method Post -ContentType 'application/json' -Body $body
```

Using curl.exe in PowerShell (note: use `curl.exe`, not the `curl` alias, and prefer forward slashes or escape backslashes):
```powershell
curl.exe -X POST "http://localhost:8000/inference" -H "Content-Type: application/json" -d "{ \"job_folder\": \"D:/path/to/job/folder\" }"
# or with backslashes escaped
curl.exe -X POST "http://localhost:8000/inference" -H "Content-Type: application/json" -d "{ \"job_folder\": \"D:\\path\\to\\job\\folder\" }"
```

Notes (Windows):
- `curl` in PowerShell is an alias; use `curl.exe` for classic curl flags.
- Use forward slashes in JSON paths, or escape backslashes.
- Bash-style line continuations (`\`) do not work in PowerShell unless using a Bash shell.

### 5. View API Documentation
Navigate to `http://localhost:8000/docs` for interactive API documentation.

## API Reference

### POST /inference
**Request Body:**
```json
{
  "job_folder": "/path/to/job/folder",
  "image_extensions": [".jpg", ".png"]  // optional
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Successfully processed 4 images",
  "total_images": 4,
  "anomalies_detected": 1,
  "average_score": 0.3245,
  "processing_time": 2.1,
  "results": [
    {
      "image_path": "/path/to/image1.jpg",
      "anomaly_score": 0.234,
      "anomaly_label": 0,
      "processing_time": 0.521
    }
  ]
}
```

## Configuration

### Server Configuration
- **Host**: Default `127.0.0.1`
- **Port**: Default `8000`
- **Model Folder**: Default `models/patchcore`
- **Workers**: Default `1` (single worker for GPU models)

### Model Configuration
- **Input Size**: 224x224 pixels
- **Normalization**: ImageNet standard
- **Providers**: DirectML → CPU fallback

## Future Development

### 🚧 Planned Components (From Original Design)
1. **YOLO Models Integration**
   - Paste detection model (`paste.pt`)
   - Pad detection model (`pad.pt`)

2. **SAM Segmentation**
   - Segment Anything Model for precise segmentation
   - Bbox-prompted segmentation

3. **Results Integration**
   - Multi-model output fusion
   - Priority-based defect classification
   - CSV output generation matching original format

4. **Go Application Integration**
   - Communication with `02.exe`
   - Job folder monitoring
   - Completion signaling

### 📈 Performance Optimizations
- [ ] GPU memory optimization
- [ ] Batch processing improvements
- [ ] Model quantization
- [ ] Async processing pipeline
- [ ] Results caching

## Development

### Project Structure
```
JQ_SPI_02_AI_API/
├── models/
│   └── patchcore/          # PatchCore ONNX models
├── data/                   # Sample data and test images
├── doc/                    # Documentation
├── patchcore_server.py     # FastAPI server
├── patchcore_inference.py  # Core inference engine
├── trigger_inference.py    # Client trigger utility
└── ai_workflow_diagram.txt # Complete system design
```

### Testing
```bash
# Start server in development mode
python patchcore_server.py --host 127.0.0.1 --port 8000

# Test with sample data
python trigger_inference.py data/sample_job_folder --wait --extensions .jpg
```

## Troubleshooting

### Common Issues
1. **Model Loading Errors**: Ensure ONNX model is in correct folder
2. **GPU Issues**: Check DirectML installation or use CPU-only mode
3. **Job Folder Not Found**: Verify folder path and permissions
4. **Server Not Ready**: Wait for model initialization (check `/health`)

### Logs
Server logs provide detailed information about:
- Model loading status
- Inference timing
- Error details
- Request processing

## License

This project is part of the JQ_SPI_02_AI industrial inspection system.
