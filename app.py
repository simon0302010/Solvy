from flask import Flask, render_template, request
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


    print(questions)

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

    print(f"Received image: {len(image_binary)} bytes")

    return {'success': True, 'message': 'Image uploaded successfully'}

if __name__ == '__main__':
    app.run(debug=True, port=port)
