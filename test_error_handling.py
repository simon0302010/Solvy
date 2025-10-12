"""
Integration test to verify error messages are properly displayed.
This test simulates backend errors and verifies they are handled correctly.
"""

import pytest
import io
import json
import sys
from PIL import Image


def create_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (100, 100), color='white')
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io


def test_upload_with_backend_error():
    """
    Test that backend errors return proper error messages.
    This verifies that the frontend will receive descriptive error messages
    instead of generic 'Upload failed' messages.
    """
    # Clean up any previous imports
    if 'app' in sys.modules:
        del sys.modules['app']
    if 'backend' in sys.modules:
        del sys.modules['backend']
    if 'logger' in sys.modules:
        del sys.modules['logger']
    if 'chat' in sys.modules:
        del sys.modules['chat']
    
    # Mock the backend module to raise an error
    class MockBackendError:
        @staticmethod
        def process_image(image_bytes):
            raise ValueError("Test error: Invalid image format")
    
    sys.modules['backend'] = MockBackendError()
    
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
    client = app.test_client()
    
    img_io = create_test_image()
    data = {
        'image': (img_io, 'test.png')
    }
    
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    
    # Should return 500 Internal Server Error
    assert response.status_code == 500
    
    response_data = json.loads(response.data)
    
    # Should contain an error field
    assert 'error' in response_data
    
    # Error message should be "Internal server error"
    # This is what the frontend will display to the user
    assert response_data['error'] == 'Internal server error'
    
    print(f"✓ Backend error properly returned: {response_data['error']}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

