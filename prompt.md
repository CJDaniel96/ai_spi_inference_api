# AI Inference System - Architecture Analysis & Recommendations

## Stage 2: AI Inference & Results (`ai_processor.py`)

### Current Architecture Overview

1. **Trigger Detection**:
   - Monitor for `.done` files in the `sfcTemp` directory structure (recursive monitoring)
   - Extract timestamp from filename (format: `{timestamp}.done`)
   - Check for corresponding CSV files in `AMR` directory (format: `{timestamp}.csv`)
   - Proceed only when both `.done` marker and CSV file are present

2. **AI Processing Pipeline**:
   - Load three AI models at startup for optimal performance
   - Process images through three inference models in coordinated fashion:

   **Model Architecture**:
   - **Anomaly Detection Model**: Patchcore + EfficientNet-B5 (threshold: 0.54, batch-capable)
   - **Segmentation Pipeline**: YOLO (paste.pt) → SAM (bbox-prompt inference, sequential-only)
   - **Distance Detection Model**: YOLO (pad.pt, batch-capable)

3. **Results Integration & Output**:
   - Calculate derived metrics and apply defect classification rules (detailed below)
   - Dual output strategy: Primary (`W:\Machine\{timestamp}\AI\`) + Secondary backup
   - Test/Production mode configuration with configurable `is_pass` values

## Critical Weaknesses Identified

### 1. **Architecture Complexity & Dependencies**
- **Sequential Dependency Chain**: YOLO paste → SAM creates synchronization bottleneck
- **Memory Management Gap**: No explicit GPU memory allocation strategy
- **Model Lifecycle**: Unclear resource cleanup and model unloading procedures

### 2. **Performance & Scalability Issues**
- **SAM Bottleneck**: Sequential SAM inference limits overall throughput
- **Resource Contention**: No GPU memory reservation for concurrent AI applications
- **Batch Processing Limitation**: Mixed batch/sequential processing creates inefficiency

### 3. **Error Handling & Reliability**
- **No Fallback Strategy**: System failure if any model fails
- **Data Consistency Risk**: File cleanup before confirming successful AI completion
- **Missing Error Recovery**: No mechanism to retry failed inferences

### 4. **Operational Risks**
- **Memory Exhaustion**: Potential GPU OOM with concurrent applications
- **Processing Delays**: Sequential SAM could create significant latency
- **Single Point of Failure**: All three models must function for system success

## Performance & Resource Estimates

### GPU Memory Requirements (CUDA Production Environment)
```
Estimated GPU Memory Usage:
- Anomaly Detection (EfficientNet-B5): ~2.5GB
- YOLO Models (paste.pt + pad.pt): ~1.2GB each
- SAM Model: ~3.5GB
- Batch Processing Buffers: ~1.5GB
Total Estimated: ~10GB VRAM
```

### Inference Time Projections
```
Per Job Folder Processing:
- Batch Anomaly Detection: ~0.8s per image batch
- YOLO Paste Detection: ~0.3s per image batch  
- SAM Sequential Processing: ~1.2s per detected bbox
- YOLO Pad Detection: ~0.3s per image batch
- Results Integration: ~0.2s

Total Estimated Time: 2.8s + (1.2s × bbox_count) per job
```

## Defect Classification Rules & Logic

### Derived Metrics Calculation
```python
# Core metric calculations from AI model outputs
pad_area = π × Width × Length / 4  # Ellipse formula for pad area
cover% = insp_area_ai(pixel) × 0.8246 × 100 / pad_area  # Coverage percentage
```

### Defect Detection Thresholds & Rules
```python
# Primary defect classification (applied in priority order)
defect_rules = {
    # 1. Volume Defects (Highest Priority)
    'volume_defect': {
        'condition': 'insp_vol outside configured thresholds',
        'thresholds': '±10 or ±20 offset ranges',
        'priority': 1
    },
    
    # 2. Height Defects  
    'height_defect': {
        'condition': 'insp_hei > 200',
        'priority': 2
    },
    
    # 3. AI Anomaly Defects
    'ai_anomaly_defect': {
        'condition': 'abnormal_score > 0.46',
        'model_output': '"FM/color" for detected anomalies',
        'priority': 3
    },
    
    # 4. Distance Defects
    'distance_defect': {
        'condition': 'min_pad_distance < 6.6 AND cover% > 180%',
        'note': 'Combined distance and coverage violation',
        'priority': 4
    },
    
    # 5. Coverage Defects (Lowest Priority)
    'coverage_defect': {
        'condition': 'cover% > 180%',
        'note': 'Standalone coverage violation',
        'priority': 5
    }
}
```

### Classification Logic Implementation
```python
def classify_defects(row):
    """Apply defect classification rules with priority ordering"""
    
    # Check volume defects (highest priority)
    if is_volume_defect(row['insp_vol']):
        return 'volume_defect', 'NG'
    
    # Check height defects
    if row['insp_hei'] > 200:
        return 'height_defect', 'NG'
    
    # Check AI anomaly defects  
    if row['abnormal_score'] > 0.46:
        return 'ai_anomaly_defect', 'NG'
    
    # Check combined distance + coverage defects
    if row['min_pad_distance'] < 6.6 and row['cover%'] > 180:
        return 'distance_defect', 'NG'
    
    # Check standalone coverage defects
    if row['cover%'] > 180:
        return 'coverage_defect', 'NG'
    
    # No defects detected
    return 'OK', 'OK'

# Apply classification to update columns
df['ai_defect_name'], df['ai_juge'] = zip(*df.apply(classify_defects, axis=1))
```

### Model-Specific Outputs
- **Anomaly Detection**: `abnormal_score` (float) + `ai_defect_name` ("FM/color" | "OK")
- **Segmentation**: `insp_area_ai(pixel)` (integer pixel count)
- **Distance Detection**: `min_pad_distance` (float, minimum distance between pads)

### Configuration Parameters
```python
# Configurable thresholds for production tuning
ANOMALY_THRESHOLD = 0.46      # Anomaly detection sensitivity
COVERAGE_THRESHOLD = 180      # Coverage percentage limit  
HEIGHT_THRESHOLD = 200        # Maximum height limit
DISTANCE_THRESHOLD = 6.6      # Minimum pad distance
VOLUME_OFFSET_RANGES = {      # Volume defect ranges
    'tight': ±10,
    'loose': ±20
}
```

## Recommended Architecture Improvements

### 1. **Enhanced Error Handling**
```python
# Implement circuit breaker pattern
class ModelCircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.last_failure_time = None
        self.recovery_timeout = recovery_timeout
```

### 2. **Resource Management Strategy**
```python
# GPU memory allocation with limits
torch.cuda.set_per_process_memory_fraction(0.7)  # Reserve 30% for other apps
```

### 3. **Optimized Processing Pipeline**
- **Parallel Model Loading**: Initialize models concurrently during startup
- **Asynchronous Processing**: Use threading for I/O operations
- **Batch Optimization**: Group SAM inferences where possible

### 4. **Monitoring & Alerting**
- **Performance Metrics**: Track inference times and memory usage
- **Health Checks**: Monitor model availability and performance
- **Alert System**: Notify on failures or performance degradation

### 5. **Data Safety Improvements**
- **Transactional Processing**: Confirm successful AI processing before cleanup
- **Backup Strategy**: Retain processed files until downstream confirmation
- **Recovery Mechanism**: Ability to reprocess failed batches

## Production Deployment Considerations

1. **Resource Allocation**: Reserve 70% GPU memory for AI inference system
2. **Concurrent Applications**: Implement resource scheduling with other AI apps
3. **Performance Monitoring**: Real-time tracking of inference times and memory usage
4. **Failure Recovery**: Automated retry mechanism with exponential backoff
5. **Configuration Management**: Environment-specific tuning for test/production modes

## Project Component Breakdown & Build Guide

### Component Architecture Overview
```
AI_PROCESSOR_PROJECT/
├── 1. File Monitor Component (file_monitor.py)
├── 2. Model Manager Component (model_manager.py) 
├── 3. Anomaly Detection Component (anomaly_detector.py)
├── 4. Segmentation Pipeline Component (segmentation_pipeline.py)
├── 5. Distance Detection Component (distance_detector.py)
├── 6. Results Processor Component (results_processor.py)
├── 7. Configuration Manager (config_manager.py)
├── 8. Error Handler & Logger (error_handler.py)
└── 9. Main AI Processor (ai_processor.py)
```

## Component-by-Component Build Guide

### 1. Configuration Manager Component
**Purpose**: Centralized configuration management for all AI models and processing parameters

**Build Steps**:
```python
# Step 1: Create config_manager.py
class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        pass
```

**Implementation Details**:
- **Dependencies**: `yaml`, `os`, `pathlib`
- **Configuration Parameters**:
  - Model paths and thresholds
  - Directory paths (sfcTemp, AMR, output)
  - Processing parameters (batch sizes, timeouts)
  - Defect classification thresholds
- **Key Methods**: `load_config()`, `get_model_config()`, `get_paths()`, `validate_config()`

### 2. Error Handler & Logger Component  
**Purpose**: Centralized error handling, logging, and monitoring

**Build Steps**:
```python
# Step 1: Create error_handler.py
import logging
from enum import Enum
from typing import Optional, Dict, Any

class ErrorLevel(Enum):
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class ErrorHandler:
    def __init__(self, log_file: str = "ai_processor.log"):
        self.setup_logging(log_file)
        self.error_counts = {}
        
    def setup_logging(self, log_file: str):
        """Configure logging with rotating file handler"""
        pass
```

**Implementation Details**:
- **Dependencies**: `logging`, `traceback`, `datetime`, `json`
- **Features**: 
  - Rotating log files
  - Error counting and alerting
  - Performance metrics tracking
  - Circuit breaker integration
- **Key Methods**: `log_error()`, `log_performance()`, `check_error_threshold()`, `reset_counters()`

### 3. File Monitor Component
**Purpose**: Monitor filesystem for trigger files and manage file operations

**Build Steps**:
```python
# Step 1: Create file_monitor.py  
import os
import time
from pathlib import Path
from typing import List, Tuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileMonitor:
    def __init__(self, config_manager, error_handler):
        self.config = config_manager
        self.error_handler = error_handler
        self.observer = Observer()
        
    def start_monitoring(self):
        """Start monitoring sfcTemp directory for .done files"""
        pass
        
    def find_csv_pairs(self) -> List[Tuple[str, str]]:
        """Find matching .done and .csv file pairs"""
        pass
```

**Implementation Details**:
- **Dependencies**: `watchdog`, `pathlib`, `os`, `glob`
- **Features**:
  - Recursive directory monitoring
  - File pair validation (.done + .csv)
  - Timestamp extraction and matching
  - File cleanup management
- **Key Methods**: `start_monitoring()`, `find_csv_pairs()`, `validate_file_pair()`, `cleanup_files()`

### 4. Model Manager Component
**Purpose**: Load, manage, and optimize AI model lifecycle and GPU memory

**Build Steps**:
```python
# Step 1: Create model_manager.py
import torch
from typing import Dict, Any, Optional
from contextlib import contextmanager

class ModelManager:
    def __init__(self, config_manager, error_handler):
        self.config = config_manager  
        self.error_handler = error_handler
        self.models = {}
        self.gpu_memory_allocated = False
        
    def initialize_models(self):
        """Load all AI models with memory management"""
        try:
            self._setup_gpu_memory()
            self._load_anomaly_model()
            self._load_segmentation_models()  
            self._load_distance_model()
        except Exception as e:
            self.error_handler.log_error("Model loading failed", e)
            
    def _setup_gpu_memory(self):
        """Configure GPU memory allocation limits"""
        torch.cuda.set_per_process_memory_fraction(0.7)
```

**Implementation Details**:
- **Dependencies**: `torch`, `ultralytics`, `segment_anything`, custom model libraries
- **Features**:
  - GPU memory management (70% allocation)
  - Model preloading and caching
  - Memory cleanup utilities  
  - Model health monitoring
- **Key Methods**: `initialize_models()`, `get_model()`, `cleanup_models()`, `check_model_health()`

### 5. Anomaly Detection Component
**Purpose**: Patchcore + EfficientNet-B5 anomaly detection with batch processing

**Build Steps**:
```python
# Step 1: Create anomaly_detector.py
import torch
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image

class AnomalyDetector:
    def __init__(self, model_manager, config_manager, error_handler):
        self.model_manager = model_manager
        self.config = config_manager
        self.error_handler = error_handler
        self.threshold = config_manager.get_anomaly_threshold()
        
    def batch_detect_anomalies(self, image_paths: List[str]) -> Dict[str, Tuple[float, str]]:
        """Process batch of images for anomaly detection"""
        try:
            model = self.model_manager.get_model('anomaly')
            images = self._load_image_batch(image_paths)
            scores = self._run_inference(model, images)
            return self._process_results(image_paths, scores)
        except Exception as e:
            self.error_handler.log_error("Anomaly detection failed", e)
            return {}
```

**Implementation Details**:
- **Dependencies**: `torch`, `torchvision`, `PIL`, `numpy`, `anomalib` (or custom patchcore)
- **Features**:
  - Batch image preprocessing
  - EfficientNet-B5 feature extraction
  - Patchcore anomaly scoring
  - Configurable sensitivity threshold (0.46)
- **Key Methods**: `batch_detect_anomalies()`, `_preprocess_images()`, `_run_inference()`, `_classify_anomalies()`

### 6. Segmentation Pipeline Component  
**Purpose**: YOLO paste detection → SAM segmentation pipeline

**Build Steps**:
```python
# Step 1: Create segmentation_pipeline.py
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class SegmentationPipeline:
    def __init__(self, model_manager, config_manager, error_handler):
        self.model_manager = model_manager
        self.config = config_manager
        self.error_handler = error_handler
        self.paste_model = None
        self.sam_predictor = None
        
    def process_batch_segmentation(self, image_paths: List[str]) -> Dict[str, int]:
        """Coordinated YOLO→SAM pipeline processing"""
        try:
            # Step 1: Batch YOLO detection
            paste_detections = self._batch_yolo_detection(image_paths)
            
            # Step 2: Sequential SAM segmentation  
            segmentation_results = {}
            for image_path, bboxes in paste_detections.items():
                if bboxes:
                    closest_bbox = self._find_closest_to_center(bboxes)
                    pixel_count = self._sam_segmentation(image_path, closest_bbox)
                    segmentation_results[image_path] = pixel_count
                    
            return segmentation_results
        except Exception as e:
            self.error_handler.log_error("Segmentation pipeline failed", e)
            return {}
```

**Implementation Details**:
- **Dependencies**: `ultralytics`, `segment-anything`, `opencv-python`, `torch`
- **Features**:
  - YOLO batch detection (paste.pt model)
  - Center-closest bbox selection
  - SAM bbox-prompt segmentation
  - Pixel counting for coverage analysis
- **Key Methods**: `process_batch_segmentation()`, `_batch_yolo_detection()`, `_sam_segmentation()`, `_find_closest_to_center()`

### 7. Distance Detection Component
**Purpose**: YOLO-based pad detection and distance calculation

**Build Steps**:
```python
# Step 1: Create distance_detector.py
import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple
from scipy.spatial.distance import cdist

class DistanceDetector:
    def __init__(self, model_manager, config_manager, error_handler):
        self.model_manager = model_manager
        self.config = config_manager  
        self.error_handler = error_handler
        
    def batch_detect_distances(self, image_paths: List[str]) -> Dict[str, float]:
        """Batch process images for pad distance detection"""
        try:
            model = self.model_manager.get_model('distance')
            distance_results = {}
            
            for batch_paths in self._create_batches(image_paths):
                batch_results = self._process_batch(model, batch_paths)
                distance_results.update(batch_results)
                
            return distance_results
        except Exception as e:
            self.error_handler.log_error("Distance detection failed", e)
            return {}
            
    def _calculate_min_distance(self, pad_centers: List[Tuple[float, float]]) -> float:
        """Calculate minimum distance between pad centers"""
        if len(pad_centers) < 2:
            return float('inf')
        distances = cdist(pad_centers, pad_centers)
        np.fill_diagonal(distances, float('inf'))
        return np.min(distances)
```

**Implementation Details**:
- **Dependencies**: `ultralytics`, `scipy`, `numpy`
- **Features**:
  - YOLO batch detection (pad.pt model)
  - Pad center calculation
  - Inter-pad distance matrix computation
  - Minimum distance extraction
- **Key Methods**: `batch_detect_distances()`, `_extract_pad_centers()`, `_calculate_min_distance()`

### 8. Results Processor Component
**Purpose**: Integrate AI results, apply defect classification, and generate outputs

**Build Steps**:
```python
# Step 1: Create results_processor.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from pathlib import Path

class ResultsProcessor:
    def __init__(self, config_manager, error_handler):
        self.config = config_manager
        self.error_handler = error_handler
        self.defect_thresholds = config_manager.get_defect_thresholds()
        
    def process_results(self, csv_path: str, ai_results: Dict[str, Any]) -> bool:
        """Integrate AI results with CSV data and apply defect classification"""
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Integrate AI model results
            df = self._integrate_ai_results(df, ai_results)
            
            # Calculate derived metrics
            df = self._calculate_derived_metrics(df)
            
            # Apply defect classification
            df = self._apply_defect_classification(df)
            
            # Generate outputs
            return self._save_results(df, csv_path)
            
        except Exception as e:
            self.error_handler.log_error("Results processing failed", e)
            return False
            
    def _apply_defect_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply priority-based defect classification rules"""
        def classify_row(row):
            # Volume defects (highest priority)
            if self._is_volume_defect(row['insp_vol']):
                return 'volume_defect', 'NG'
            # ... rest of classification logic
        
        df[['ai_defect_name', 'ai_juge']] = df.apply(
            lambda row: classify_row(row), axis=1, result_type='expand'
        )
        return df
```

**Implementation Details**:
- **Dependencies**: `pandas`, `numpy`, `pathlib`
- **Features**:
  - AI results integration with CSV data
  - Derived metrics calculation (pad_area, cover%)
  - Priority-based defect classification
  - Dual output strategy (primary + backup)
- **Key Methods**: `process_results()`, `_integrate_ai_results()`, `_apply_defect_classification()`, `_save_results()`

### 9. Main AI Processor Component
**Purpose**: Orchestrate all components and manage the processing workflow

**Build Steps**:
```python
# Step 1: Create ai_processor.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

class AIProcessor:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.error_handler = ErrorHandler()
        self.model_manager = ModelManager(self.config_manager, self.error_handler)
        self.file_monitor = FileMonitor(self.config_manager, self.error_handler)
        
        # AI Components
        self.anomaly_detector = AnomalyDetector(self.model_manager, self.config_manager, self.error_handler)
        self.segmentation_pipeline = SegmentationPipeline(self.model_manager, self.config_manager, self.error_handler)
        self.distance_detector = DistanceDetector(self.model_manager, self.config_manager, self.error_handler)
        self.results_processor = ResultsProcessor(self.config_manager, self.error_handler)
        
    async def run(self):
        """Main processing loop"""
        await self._initialize_system()
        await self._start_processing_loop()
        
    async def _process_job_batch(self, file_pairs: List[Tuple[str, str]]):
        """Process batch of CSV files with AI inference"""
        for done_file, csv_file in file_pairs:
            try:
                # Extract image paths from CSV
                image_paths = self._extract_image_paths(csv_file)
                
                # Run AI models in parallel where possible
                ai_results = await self._run_ai_inference(image_paths)
                
                # Process results and generate outputs
                success = self.results_processor.process_results(csv_file, ai_results)
                
                if success:
                    self.file_monitor.cleanup_files(done_file, csv_file)
                    
            except Exception as e:
                self.error_handler.log_error(f"Job processing failed: {csv_file}", e)
```

**Implementation Details**:
- **Dependencies**: `asyncio`, `concurrent.futures`, all component modules
- **Features**:
  - Asynchronous processing coordination
  - Parallel AI model execution where possible
  - Error handling and recovery
  - Performance monitoring
- **Key Methods**: `run()`, `_process_job_batch()`, `_run_ai_inference()`, `_initialize_system()`

## Build Order & Dependencies

### Phase 1: Foundation (Build First)
1. **Configuration Manager** - No dependencies
2. **Error Handler & Logger** - Depends on: Configuration Manager

### Phase 2: Core Infrastructure  
3. **File Monitor** - Depends on: Config Manager, Error Handler
4. **Model Manager** - Depends on: Config Manager, Error Handler

### Phase 3: AI Components (Can build in parallel)
5. **Anomaly Detector** - Depends on: Model Manager, Config Manager, Error Handler
6. **Distance Detector** - Depends on: Model Manager, Config Manager, Error Handler  
7. **Segmentation Pipeline** - Depends on: Model Manager, Config Manager, Error Handler

### Phase 4: Integration
8. **Results Processor** - Depends on: Config Manager, Error Handler
9. **Main AI Processor** - Depends on: All components

## Testing Strategy per Component

### Unit Testing
- Each component should have isolated unit tests
- Mock dependencies for testing individual components
- Test error conditions and edge cases

### Integration Testing  
- Test component interactions (e.g., Model Manager ↔ AI Components)
- Test file processing workflows
- Test GPU memory management under load

### End-to-End Testing
- Complete workflow testing with sample data
- Performance benchmarking with realistic datasets
- Error recovery and cleanup testing

## Conclusion

While the current architecture provides comprehensive AI analysis capabilities, it requires significant improvements in error handling, resource management, and performance optimization before production deployment. The sequential SAM dependency and lack of robust error handling present the highest risks to system reliability and performance.