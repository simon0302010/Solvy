import pytest
import io
import json
from PIL import Image


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    # Import here to avoid circular dependencies
    import sys
    import os
    
    # Mock the backend module before importing app
    class MockBackend:
        @staticmethod
        def process_image(image_bytes):
            import numpy as np
            # Return a simple mock processed image
            return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    sys.modules['backend'] = MockBackend()
    
    # Mock other dependencies
    class MockLogger:
        @staticmethod
        def print_info(msg):
            pass
        
        @staticmethod
        def print_warn(msg):
            pass
    
    sys.modules['logger'] = MockLogger()
    
    class MockChat:
        @staticmethod
        def chat_with_gemini(chat_id, user_input, image_base64):
            return "Mock response"
        
        @staticmethod
        def get_chat_history(chat_id):
            return []
    
    sys.modules['chat'] = MockChat()
    
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def create_test_image():
    """Create a simple test image."""
    # Create a simple 100x100 white image
    img = Image.new('RGB', (100, 100), color='white')
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io


def test_upload_no_file(client):
    """Test upload endpoint with no file."""
    response = client.post('/upload')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No image file'


def test_upload_empty_filename(client):
    """Test upload endpoint with empty filename."""
    data = {
        'image': (io.BytesIO(b''), '')
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No file selected'


def test_upload_valid_image(client):
    """Test upload endpoint with a valid image."""
    img_io = create_test_image()
    data = {
        'image': (img_io, 'test.png')
    }
    
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    
    response_data = json.loads(response.data)
    assert 'image_data' in response_data
    assert 'id' in response_data
    assert 'filename' in response_data
    assert response_data['filename'] == 'test.png'
    assert response_data['image_data'].startswith('data:image/png;base64,')


def test_home_route(client):
    """Test home route."""
    response = client.get('/')
    assert response.status_code == 200


def test_scan_route(client):
    """Test scan route."""
    response = client.get('/scan')
    assert response.status_code == 200


def test_chat_root_route(client):
    """Test chat root route."""
    response = client.get('/chat')
    assert response.status_code == 200


def test_settings_route(client):
    """Test settings route."""
    response = client.get('/settings')
    assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

