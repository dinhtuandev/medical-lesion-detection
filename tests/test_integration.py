"""
Integration Tests for Complete Pipeline
Tests preprocessing + prediction flow
"""

import pytest
import numpy as np
import cv2
import os
from src.preprocess import preprocess_image
from src.predict import run_prediction


class TestFullPipeline:
    """Test complete preprocessing -> prediction pipeline"""
    
    @pytest.fixture
    def test_xray_image(self):
        """Create a realistic test X-ray image"""
        # Create a test image that simulates an X-ray
        img = np.full((800, 600, 3), 220, dtype=np.uint8)  # Light gray background
        
        # Add some anatomical features
        img[100:700, 150:500] = 150  # Lung area
        img[300:400, 200:350] = 100  # Potential lesion area
        
        temp_path = '/tmp/test_xray.jpg'
        cv2.imwrite(temp_path, img)
        yield temp_path
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_preprocess_then_predict(self, test_xray_image):
        """Test full pipeline: preprocess -> predict"""
        
        # Step 1: Preprocess
        processed = preprocess_image(test_xray_image, size=(640, 640))
        assert processed is not None
        assert processed.shape == (640, 640, 3)
        
        # Step 2: Save processed image temporarily
        temp_processed = '/tmp/processed_xray.jpg'
        cv2.imwrite(temp_processed, processed)
        
        # Step 3: Predict
        results = run_prediction('models/yolov8_best.pt', temp_processed)
        
        # Step 4: Verify results
        assert results is not None
        assert len(results) > 0
        assert hasattr(results[0], 'boxes')
        
        # Cleanup
        os.remove(temp_processed)

    def test_pipeline_with_real_test_data(self):
        """Test pipeline with actual test data from data/test"""
        test_dir = 'data/test/images'
        
        if os.path.exists(test_dir):
            test_images = [f for f in os.listdir(test_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            if len(test_images) > 0:
                # Test first 3 images
                for img_file in test_images[:3]:
                    img_path = os.path.join(test_dir, img_file)
                    
                    # Preprocess
                    processed = preprocess_image(img_path)
                    assert processed is not None
                    
                    # Save and predict
                    temp_path = f'/tmp/test_{img_file}'
                    cv2.imwrite(temp_path, processed)
                    
                    results = run_prediction('models/yolov8_best.pt', temp_path)
                    assert results is not None
                    
                    os.remove(temp_path)


class TestPipelineQuality:
    """Test quality of preprocessed images before prediction"""
    
    def test_preprocessed_image_not_corrupted(self):
        """Test that preprocessing doesn't corrupt image"""
        # Create test image
        original = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
        temp_original = '/tmp/original_test.jpg'
        cv2.imwrite(temp_original, original)
        
        # Preprocess
        processed = preprocess_image(temp_original)
        
        # Check image properties
        assert processed is not None
        assert processed.dtype == np.uint8
        assert len(processed.shape) == 3
        assert processed.shape[2] == 3
        assert processed.min() >= 0
        assert processed.max() <= 255
        
        os.remove(temp_original)

    def test_clahe_enhancement_effect(self):
        """Test that CLAHE actually enhances contrast in pipeline"""
        # Create low-contrast image
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        img[100:500, 100:500, :] = 120  # Subtle difference
        
        temp_path = '/tmp/low_contrast.jpg'
        cv2.imwrite(temp_path, img)
        
        # Preprocess (applies CLAHE)
        processed = preprocess_image(temp_path)
        
        # Check that contrast increased
        # (darker areas should be darker, lighter areas lighter)
        original_std = np.std(img)
        processed_std = np.std(processed)
        
        assert processed_std > original_std, "CLAHE should increase contrast"
        
        os.remove(temp_path)

    def test_preprocessing_handles_different_lighting(self):
        """Test preprocessing works with different lighting conditions"""
        
        # Create dark image
        dark_img = np.full((640, 640, 3), 50, dtype=np.uint8)
        temp_dark = '/tmp/dark_test.jpg'
        cv2.imwrite(temp_dark, dark_img)
        
        # Create bright image
        bright_img = np.full((640, 640, 3), 200, dtype=np.uint8)
        temp_bright = '/tmp/bright_test.jpg'
        cv2.imwrite(temp_bright, bright_img)
        
        # Process both
        processed_dark = preprocess_image(temp_dark)
        processed_bright = preprocess_image(temp_bright)
        
        # Both should be valid
        assert processed_dark is not None
        assert processed_bright is not None
        
        # After CLAHE, they should be more similar (normalized contrast)
        # Original difference is large
        original_diff = np.abs(dark_img.mean() - bright_img.mean())
        # Processed difference should be smaller due to adaptive enhancement
        processed_diff = np.abs(processed_dark.mean() - processed_bright.mean())
        
        # CLAHE helps level out extreme lighting
        assert processed_diff < original_diff
        
        os.remove(temp_dark)
        os.remove(temp_bright)


class TestPipelineRobustness:
    """Test pipeline robustness to various inputs"""
    
    def test_pipeline_with_various_sizes(self):
        """Test pipeline works with various image sizes"""
        sizes = [(256, 256), (512, 512), (800, 600), (1024, 768)]
        
        for h, w in sizes:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            temp_path = f'/tmp/size_test_{h}x{w}.jpg'
            cv2.imwrite(temp_path, img)
            
            # Should resize to 640x640 regardless of input size
            processed = preprocess_image(temp_path, size=(640, 640))
            
            assert processed is not None
            assert processed.shape == (640, 640, 3)
            
            os.remove(temp_path)

    def test_pipeline_with_different_formats(self):
        """Test pipeline with different image formats"""
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        
        # Test JPG
        jpg_path = '/tmp/test_image.jpg'
        cv2.imwrite(jpg_path, img)
        result_jpg = preprocess_image(jpg_path)
        assert result_jpg is not None
        os.remove(jpg_path)
        
        # Test PNG
        png_path = '/tmp/test_image.png'
        cv2.imwrite(png_path, img)
        result_png = preprocess_image(png_path)
        assert result_png is not None
        os.remove(png_path)

    def test_consecutive_pipeline_runs(self):
        """Test pipeline can be run multiple times consecutively"""
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        temp_path = '/tmp/consecutive_test.jpg'
        cv2.imwrite(temp_path, img)
        
        # Run pipeline 5 times
        results = []
        for i in range(5):
            processed = preprocess_image(temp_path)
            assert processed is not None
            results.append(processed)
        
        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i])
        
        os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
