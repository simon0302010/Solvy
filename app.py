from flask import Flask, render_template, request
from logger import *
import base64
import re
import os

try:
    port = os.getenv("PORT")
except:
    port = 5000

app = Flask(__name__)

@app.route('/')
def home():
    # Sample data to simulate scanned questions
    questions = [{
        'type': 'Question Title',
        'status': 'Status',
        'description': 'You scanned a question. The AI solved it with this discription.'
    }] * 4  # Repeat 4 times for demo

    return render_template('home.html', questions=questions)

@app.route('/scan')
def scan():
    return render_template('scan.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return {'error': 'No image file'}, 400
    
    file = request.files['image']
    if file.filename == '':
        return {'error': 'No file selected'}, 400

    image_binary = file.read()

    with open("image.png", "wb") as f:
        f.write(image_binary)

    print_info(f"Received image: {len(image_binary)} bytes")

    return {'success': True, 'message': 'Image uploaded successfully'}

if __name__ == '__main__':
    app.run(debug=True, port=port, host="0.0.0.0")
