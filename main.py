from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Sample data to simulate scanned questions
    questions = [{
        'type': 'Algebra Question',
        'status': 'Solved',
        'description': 'You scanned an algebraic equation. The AI solved it and found x = 5.'
    }] * 4  # Repeat 4 times for demo

    return render_template('home.html', questions=questions)

if __name__ == '__main__':
    app.run(debug=True)
