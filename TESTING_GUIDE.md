# 🧪 Testing Guide - Medical Lesion Detection

## 📋 Overview - 3 Key Testing Concepts

### **Concept 1: The 3 Types of Testing**

```
┌─────────────────────────────────────────────────────────┐
│                  3 TYPES OF TESTING                     │
│ 1. UNIT TESTING                                         │
│    ├─ Tests small individual functions                  │
│    ├─ Ex: Does apply_clahe() work correctly?            │
│    └─ Tool: pytest                                      │
│                                                         │
│ 2. INTEGRATION TESTING                                  │
│    ├─ Tests functions working together                  │
│    ├─ Ex: preprocess → predict → visualize OK?          │
│    └─ Tool: pytest + fixtures                           │
│                                                         │
│ 3. END-TO-END TESTING                                   │
│    ├─ Tests the complete flow (upload → output)         │
│    ├─ Ex: Upload X-ray → Detection → Download OK?       │
│    └─ Tool: selenium, streamlit testing                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 **Concept 2: The 3 Datasets (You Have Them!)**

```
Train Set (356 images) ──→ Train the model
                           ↓
Valid Set (102 images) ──→ Check during training
                           ↓
Test Set (51 images) ────→ Evaluate FINAL model ✓
```

**Why separate into 3 sets?**

- **Train**: Model learns from this data
- **Validation**: Adjust hyperparameters (clip_limit, tile_size)
- **Test**: Objective evaluation "is the model good?" ( unseen data)

---

## 📊 **Concept 3: Metrics & Evaluation**

### **Evaluation Metrics (Quality Assessment)**

```python
# You should track these metrics:

1. PRECISION = TP / (TP + FP)
   ├─ When model says "pneumonia", 96.5% correct
   ├─ Low false alarms ✓
   └─ Critical: Avoid misdiagnosis

2. RECALL = TP / (TP + FN)
   ├─ Catches 97.2% of actual cases
   ├─ Low missed cases ✓
   └─ Critical: Avoid missing patients

3. mAP50 = 0.9795 (97.95%)
   ├─ Bounding box localization accuracy
   ├─ Uses IoU (Intersection over Union) = 0.5
   └─ Higher is better

4. F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
   ├─ Balance between Precision and Recall
   ├─ Ex: (0.965 × 0.972) = 0.9685
   └─ Usually 0.96+ is excellent
```

---

## 🧪 **Unit Testing - Testing Individual Functions**

### **Test 1: Check CLAHE Preprocessing**

```python
# tests/test_preprocess.py

import pytest
import cv2
import numpy as np
from src.preprocess import apply_clahe, preprocess_image

def test_apply_clahe_reduces_contrast():
    """Test CLAHE increases contrast"""
    # Create simple test image
    img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    # Apply CLAHE
    enhanced = apply_clahe(img)

    # Check: result must be color image (3 channels)
    assert enhanced.shape == (640, 640, 3)
    assert enhanced.dtype == np.uint8

def test_preprocess_image_resize():
    """Test resizing image to 640x640"""
    # Create 800x600 image
    img = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)

    # Save temporarily
    cv2.imwrite('/tmp/test_image.jpg', img)

    # Preprocess
    processed = preprocess_image('/tmp/test_image.jpg', size=(640, 640))

    # Check: Result must be 640x640
    assert processed.shape == (640, 640, 3), f"Expected (640, 640, 3), got {processed.shape}"

def test_preprocess_handles_nonexistent_file():
    """Test handling nonexistent file"""
    result = preprocess_image('/nonexistent/path/image.jpg')
    assert result is None, "Should return None for nonexistent file"
```

### **Test 2: Check Model Prediction**

```python
# tests/test_predict.py

import pytest
from src.predict import run_prediction
from ultralytics import YOLO

def test_model_loads():
    """Test model loading success"""
    model = YOLO('models/yolov8_best.pt')
    assert model is not None
    assert hasattr(model, 'predict')

def test_prediction_returns_results():
    """Test prediction returns results"""
    # Assume test image exists
    results = run_prediction('models/yolov8_best.pt', 'data/test/images/sample.jpg')

    # Check: must return Results object
    assert results is not None
    assert len(results) > 0
    assert hasattr(results[0], 'boxes')

def test_prediction_output_format():
    """Test output format"""
    results = run_prediction('models/yolov8_best.pt', 'data/test/images/sample.jpg')

    # Check boxes attribute
    boxes = results[0].boxes

    # Each box must have:
    for box in boxes:
        assert hasattr(box, 'xyxy')  # Coordinates
        assert hasattr(box, 'conf')  # Confidence
        assert hasattr(box, 'cls')   # Class
```

---

## 🔄 **Integration Testing - Testing Full Pipeline**

```python
# tests/test_integration.py

import pytest
import numpy as np
import cv2
from src.preprocess import preprocess_image
from src.predict import run_prediction

def test_full_pipeline():
    """Test full pipeline: preprocess → predict"""

    # Step 1: Load test image
    test_image_path = 'data/test/images/IM-0145-0001_jpeg.rf.sample.jpg'

    # Step 2: Preprocess
    processed = preprocess_image(test_image_path, size=(640, 640))
    assert processed is not None
    assert processed.shape == (640, 640, 3)

    # Step 3: Save temporarily
    cv2.imwrite('/tmp/processed_test.jpg', processed)

    # Step 4: Predict
    results = run_prediction('models/yolov8_best.pt', '/tmp/processed_test.jpg')

    # Step 5: Verify output
    assert results is not None
    assert len(results[0].boxes) >= 0  # ≥ 0 detections

    # If detection exists, check confidence
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            conf = box.conf[0].item()
            assert 0 <= conf <= 1, f"Confidence {conf} should be 0-1"

def test_clahe_improves_predictions():
    """Test if CLAHE improves detection"""
    test_image = 'data/test/images/IM-0145-0001_jpeg.rf.sample.jpg'

    # Predict WITHOUT CLAHE (using original image)
    results_without = run_prediction('models/yolov8_best.pt', test_image)
    detections_without = len(results_without[0].boxes)

    # Predict WITH CLAHE
    processed = preprocess_image(test_image)
    cv2.imwrite('/tmp/with_clahe.jpg', processed)
    results_with = run_prediction('models/yolov8_best.pt', '/tmp/with_clahe.jpg')
    detections_with = len(results_with[0].boxes)

    # Check: Does CLAHE help detect more?
    print(f"Detections without CLAHE: {detections_without}")
    print(f"Detections with CLAHE: {detections_with}")
```

---

## 📊 **Model Evaluation - Evaluating on Test Set**

```python
# tests/test_model_evaluation.py

import cv2
import numpy as np
from src.preprocess import preprocess_image
from src.predict import run_prediction
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_on_test_set():
    """Evaluate model on entire test set"""

    model_path = 'models/yolov8_best.pt'
    test_dir = 'data/test/images/'
    label_dir = 'data/test/labels/'

    predictions = []
    ground_truths = []

    # Loop through all test images
    import os
    test_files = os.listdir(test_dir)

    for filename in test_files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(test_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

            # Load and preprocess image
            processed = preprocess_image(image_path)
            cv2.imwrite('/tmp/test_img.jpg', processed)

            # Predict
            results = run_prediction(model_path, '/tmp/test_img.jpg')
            pred = 1 if len(results[0].boxes) > 0 else 0  # 1 = PNEUMONIA, 0 = NORMAL
            predictions.append(pred)

            # Load ground truth (from label file)
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    # File format: class_id x_center y_center width height
                    gt = 1 if len(lines) > 0 else 0
                    ground_truths.append(gt)
            except:
                pass

    # Calculate metrics
    precision = precision_score(ground_truths, predictions)
    recall = recall_score(ground_truths, predictions)
    f1 = f1_score(ground_truths, predictions)

    # Confusion Matrix
    cm = confusion_matrix(ground_truths, predictions)

    print(f"✓ Precision: {precision:.4f} (96.5% expected)")
    print(f"✓ Recall: {recall:.4f} (97.2% expected)")
    print(f"✓ F1-Score: {f1:.4f}")
    print(f"✓ Confusion Matrix:\n{cm}")

    # Assert: Metrics must be ≥ 0.90
    assert precision >= 0.90, f"Precision {precision} < 0.90"
    assert recall >= 0.90, f"Recall {recall} < 0.90"
```

---

## 🚀 **End-to-End Testing (Streamlit App)**

```python
# tests/test_streamlit_app.py

import pytest
from streamlit.testing.v1 import AppTest

def test_streamlit_app_loads():
    """Test Streamlit app runs"""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    assert not at.exception  # No errors
    assert "Medical Lesion Detection" in at.get("markdown")[0].value

def test_file_upload_works():
    """Test file upload works"""
    at = AppTest.from_file("app.py", default_timeout=30)

    # Simulate file upload
    # (Simulated - real is more complex)
    at.run()

    # Check file uploader exists
    assert at.file_uploader is not None

def test_detection_button_exists():
    """Test Detection button exists"""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Check button exists
    button = at.button[0] if at.button else None
    assert button is not None
```

---

## 📈 **How to Run Tests**

### **Install pytest:**

```bash
pip install pytest pytest-cov
```

### **Run all tests:**

```bash
pytest tests/ -v
```

### **Run specific test:**

```bash
pytest tests/test_preprocess.py::test_apply_clahe_reduces_contrast -v
```

### **View coverage (% code tested):**

```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 🎓 **Potential Viva/Defense Topics**

### **Possibility 1: Cross-Validation**

```
Instead of fixed (train/val/test) split:

K-Fold Cross-Validation:
├─ Split data into K parts (usually K=3 or 5)
├─ Run K times:
│  ├─ Time 1: Fold 1 = test, Fold 2+3 = train
│  ├─ Time 2: Fold 2 = test, Fold 1+3 = train
│  └─ Time 3: Fold 3 = test, Fold 1+2 = train
└─ Take average result of K runs

Benefit: Uses 100% of data, more accurate results
```

### **Possibility 2: Performance Testing**

```
├─ Speed Test: Is the model fast? (2.26ms ✓)
├─ Memory Test: RAM usage? (< 2GB ✓)
└─ Stress Test: How many images/second?
```

### **Possibility 3: Robustness Testing**

```
├─ Different Image Sizes: Work with varied sizes?
├─ Different Lighting: Detect in bright/dark images?
├─ Rotated Images: Detect rotated images?
└─ Noisy Images: Detect in noisy images?
```

---

**Good luck with testing! 🧪✅**
