"""
Unit Tests for Image Preprocessing Module
Tests CLAHE and image resizing functionality
"""

import pytest
import numpy as np
import cv2
import os
from src.preprocess import apply_clahe, preprocess_image


class TestCLAHE:
    """Test CLAHE preprocessing"""
    
    def test_apply_clahe_grayscale_input(self):
        """Test CLAHE on grayscale image"""
        # Create a random grayscale image
        gray_img = np.random.randint(0, 256, (640, 640), dtype=np.uint8)
        
        # Apply CLAHE (will convert to BGR)
        result = apply_clahe(gray_img)
        
        # Check output is BGR (3 channels)
        assert result.shape == (640, 640, 3), f"Expected (640, 640, 3), got {result.shape}"
        assert result.dtype == np.uint8

    def test_apply_clahe_bgr_input(self):
        """Test CLAHE on BGR color image"""
        # Create a random BGR image
        bgr_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        
        # Apply CLAHE
        result = apply_clahe(bgr_img)
        
        # Check output shape and type
        assert result.shape == (640, 640, 3)
        assert result.dtype == np.uint8

    def test_clahe_increases_contrast(self):
        """Test that CLAHE actually increases local contrast"""
        # Create a low-contrast image
        base = np.full((100, 100, 3), 128, dtype=np.uint8)
        low_contrast = base.copy()
        low_contrast[30:70, 30:70] = 100  # Dark square
        
        # Apply CLAHE
        enhanced = apply_clahe(low_contrast)
        
        # Check that some values changed (contrast increased)
        assert not np.array_equal(low_contrast, enhanced)

    def test_clahe_output_range(self):
        """Test CLAHE output is in valid uint8 range"""
        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = apply_clahe(img)
        
        assert result.min() >= 0
        assert result.max() <= 255


class TestPreprocessImage:
    """Test full image preprocessing pipeline"""
    
    @pytest.fixture
    def temp_test_image(self):
        """Create a temporary test image"""
        test_img = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
        temp_path = '/tmp/test_xray_image.jpg'
        cv2.imwrite(temp_path, test_img)
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_preprocess_image_resize(self, temp_test_image):
        """Test that preprocess resizes to 640x640"""
        result = preprocess_image(temp_test_image, size=(640, 640))
        
        assert result is not None
        assert result.shape == (640, 640, 3)

    def test_preprocess_custom_size(self, temp_test_image):
        """Test preprocess with custom size"""
        result = preprocess_image(temp_test_image, size=(512, 512))
        
        assert result is not None
        assert result.shape == (512, 512, 3)

    def test_preprocess_nonexistent_file(self):
        """Test preprocess handles nonexistent file gracefully"""
        result = preprocess_image('/nonexistent/path/image.jpg')
        
        assert result is None, "Should return None for nonexistent file"

    def test_preprocess_returns_bgr(self, temp_test_image):
        """Test preprocessed image is in BGR format"""
        result = preprocess_image(temp_test_image)
        
        assert result is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # 3 channels (BGR)
        assert result.dtype == np.uint8

    def test_preprocess_preserves_data_type(self, temp_test_image):
        """Test that preprocessing preserves uint8 data type"""
        result = preprocess_image(temp_test_image)
        
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255


class TestPreprocessIntegration:
    """Integration tests for preprocessing"""
    
    def test_clahe_then_resize_pipeline(self):
        """Test full pipeline: CLAHE -> Resize"""
        # Create test image
        test_img = np.random.randint(50, 200, (800, 600, 3), dtype=np.uint8)
        temp_path = '/tmp/pipeline_test.jpg'
        cv2.imwrite(temp_path, test_img)
        
        # Run preprocessing
        result = preprocess_image(temp_path, size=(640, 640))
        
        # Verify
        assert result is not None
        assert result.shape == (640, 640, 3)
        assert result.dtype == np.uint8
        
        # Cleanup
        os.remove(temp_path)

    def test_preprocess_handles_various_dimensions(self):
        """Test preprocessing works with various input dimensions"""
        dimensions = [(640, 480, 3), (1024, 768, 3), (256, 256, 3)]
        target_size = (640, 640)
        
        for dim in dimensions:
            test_img = np.random.randint(0, 256, dim, dtype=np.uint8)
            temp_path = f'/tmp/test_{dim[0]}x{dim[1]}.jpg'
            cv2.imwrite(temp_path, test_img)
            
            result = preprocess_image(temp_path, size=target_size)
            
            assert result is not None
            assert result.shape == (*target_size, 3)
            
            os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
