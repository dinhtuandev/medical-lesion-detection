"""
Unit Tests for Model Prediction Module
Tests YOLOv8 model loading and inference
"""

import pytest
import numpy as np
import cv2
import os
from src.predict import run_prediction
from ultralytics import YOLO


class TestModelLoading:
    """Test model loading functionality"""
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        model_path = 'models/yolov8_best.pt'
        assert os.path.exists(model_path), f"Model file not found at {model_path}"

    def test_yolo_model_loads(self):
        """Test YOLO model loads successfully"""
        model = YOLO('models/yolov8_best.pt')
        assert model is not None
        assert hasattr(model, 'predict')

    def test_model_has_required_attributes(self):
        """Test model has required attributes"""
        model = YOLO('models/yolov8_best.pt')
        
        # Check model attributes
        assert hasattr(model, 'conf')
        assert hasattr(model, 'iou')
        assert hasattr(model, 'predict')


class TestPredictionOutput:
    """Test prediction output format"""
    
    @pytest.fixture
    def sample_test_image(self):
        """Create a sample test image"""
        test_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        temp_path = '/tmp/test_xray_for_pred.jpg'
        cv2.imwrite(temp_path, test_img)
        yield temp_path
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_prediction_returns_results_object(self, sample_test_image):
        """Test prediction returns Results object"""
        results = run_prediction('models/yolov8_best.pt', sample_test_image)
        
        assert results is not None
        assert len(results) > 0

    def test_results_have_boxes_attribute(self, sample_test_image):
        """Test results contain boxes attribute"""
        results = run_prediction('models/yolov8_best.pt', sample_test_image)
        
        assert hasattr(results[0], 'boxes')
        assert results[0].boxes is not None

    def test_boxes_have_required_fields(self, sample_test_image):
        """Test each box has xyxy, conf, and cls"""
        results = run_prediction('models/yolov8_best.pt', sample_test_image)
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                assert hasattr(box, 'xyxy')
                assert hasattr(box, 'conf')
                assert hasattr(box, 'cls')

    def test_confidence_scores_in_valid_range(self, sample_test_image):
        """Test confidence scores are between 0 and 1"""
        results = run_prediction('models/yolov8_best.pt', sample_test_image)
        
        for box in results[0].boxes:
            conf = box.conf[0].item()
            assert 0 <= conf <= 1, f"Confidence {conf} should be 0-1"

    def test_class_index_valid(self, sample_test_image):
        """Test class indices are valid (0 or 1)"""
        results = run_prediction('models/yolov8_best.pt', sample_test_image)
        
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            assert cls in [0, 1], f"Class {cls} should be 0 (NORMAL) or 1 (PNEUMONIA)"

    def test_bounding_box_coordinates_valid(self, sample_test_image):
        """Test bounding box coordinates are within image dimensions"""
        results = run_prediction('models/yolov8_best.pt', sample_test_image)
        
        # Image is 640x640
        max_coord = 640
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            assert 0 <= x1 <= max_coord
            assert 0 <= y1 <= max_coord
            assert 0 <= x2 <= max_coord
            assert 0 <= y2 <= max_coord
            assert x1 < x2, "x1 should be < x2"
            assert y1 < y2, "y1 should be < y2"


class TestPredictionConsistency:
    """Test prediction consistency"""
    
    @pytest.fixture
    def consistent_test_image(self):
        """Create a consistent test image"""
        # Create a deterministic image (not random)
        test_img = np.full((640, 640, 3), 128, dtype=np.uint8)
        # Add a dark square (simulating pneumonia region)
        test_img[200:400, 200:400] = 50
        
        temp_path = '/tmp/consistent_test.jpg'
        cv2.imwrite(temp_path, test_img)
        yield temp_path
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_same_image_produces_same_results(self, consistent_test_image):
        """Test that same image produces consistent predictions"""
        # Run prediction twice on same image
        results1 = run_prediction('models/yolov8_best.pt', consistent_test_image)
        results2 = run_prediction('models/yolov8_best.pt', consistent_test_image)
        
        # Should have same number of detections
        assert len(results1[0].boxes) == len(results2[0].boxes)
        
        # Should have similar confidence scores
        if len(results1[0].boxes) > 0:
            for box1, box2 in zip(results1[0].boxes, results2[0].boxes):
                conf1 = box1.conf[0].item()
                conf2 = box2.conf[0].item()
                # Allow small floating point differences
                assert abs(conf1 - conf2) < 0.001


class TestPredictionEdgeCases:
    """Test edge cases"""
    
    def test_all_black_image(self):
        """Test prediction on all-black image"""
        black_img = np.zeros((640, 640, 3), dtype=np.uint8)
        temp_path = '/tmp/black_image.jpg'
        cv2.imwrite(temp_path, black_img)
        
        results = run_prediction('models/yolov8_best.pt', temp_path)
        
        assert results is not None
        # Model should handle black image (may or may not detect)
        assert len(results[0].boxes) >= 0
        
        os.remove(temp_path)

    def test_all_white_image(self):
        """Test prediction on all-white image"""
        white_img = np.full((640, 640, 3), 255, dtype=np.uint8)
        temp_path = '/tmp/white_image.jpg'
        cv2.imwrite(temp_path, white_img)
        
        results = run_prediction('models/yolov8_best.pt', temp_path)
        
        assert results is not None
        assert len(results[0].boxes) >= 0
        
        os.remove(temp_path)

    def test_high_contrast_image(self):
        """Test prediction on high-contrast image"""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[100:500, 100:500] = 255  # White square on black
        
        temp_path = '/tmp/contrast_image.jpg'
        cv2.imwrite(temp_path, img)
        
        results = run_prediction('models/yolov8_best.pt', temp_path)
        
        assert results is not None
        assert len(results[0].boxes) >= 0
        
        os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
