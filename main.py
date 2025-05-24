from flask import Flask, render_template

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

if __name__ == '__main__':
    app.run(debug=True)
