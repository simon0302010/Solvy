from flask import Flask, render_template, request, jsonify
from logger import *
import cv2
import backend
import base64
import uuid
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session-based features

@app.route('/')
def home():
    # Sample questions (static for now)
    questions = [{
        'type': 'Algebra - Solving Equations',
        'status': 'Solved',
        'description': 'You scanned a worksheet: Solve for x in the equation 3x - 7 = 11.'
    }, {
        'type': 'Geometry - Angles in Triangles',
        'status': 'Solved',
        'description': 'You scanned a worksheet: Find the missing angle in a triangle if the other two are 45° and 65°.'
    }, {
        'type': 'Linear Equations - Word Problem',
        'status': 'Solved',
        'description': 'Sarah is twice as old as John. Together, their ages add up to 36. How old is each?'
    }, {
        'type': 'Quadratic Equations - Factoring',
        'status': 'Solved',
        'description': 'You scanned a question: Factor the quadratic equation x² - 5x + 6 = 0.'
    }]

    return render_template('home.html', questions=questions)

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    image_bytes = file.read()
    print_info(f"Received image: {len(image_bytes)} bytes")
    print_info("Processing image...")

    try:
        # Process the uploaded image with your backend logic
        processed_image = backend.process_image(image_bytes)

        # Encode to PNG and then to base64
        success, encoded_image = cv2.imencode('.png', processed_image)
        if not success:
            return jsonify({'error': 'Failed to encode image'}), 500

        base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        data_uri = f"data:image/png;base64,{base64_image}"

        # Optionally track with UUID (not required unless for analytics)
        image_id = str(uuid.uuid4())

        return jsonify({'image_data': data_uri, 'id': image_id})
    
    except Exception as e:
        print_error(f"Error processing image: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")
