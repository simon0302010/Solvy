from flask import Flask, render_template, request
import base64
import re

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

@app.route('/upload')
def upload():
    data = request.get_json()
    image_data = data['image']
    print(image_data)
    # Remove the header part ("data:image/png;base64,")
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    image_binary = base64.b64decode(image_data)

if __name__ == '__main__':
    app.run(debug=True)
