import base64
import os
import uuid

import cv2
from flask import Flask, jsonify, render_template, request

import backend
from logger import print_info, print_warn

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session-based features


@app.route("/")
def home():
    # History will now be loaded from localStorage via JavaScript
    # This is just a fallback for when localStorage is empty
    questions = []
    return render_template("home.html", questions=questions)


@app.route("/scan")
def scan():
    return render_template("scan.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    print_info(f"Received image: {len(image_bytes)} bytes")
    print_info("Processing image...")

    try:
        # Process the uploaded image with your backend logic
        processed_image = backend.process_image(image_bytes)

        # Encode to PNG and then to base64
        success, encoded_image = cv2.imencode(".png", processed_image)
        if not success:
            return jsonify({"error": "Failed to encode image"}), 500

        base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_image}"

        # Generate UUID for tracking
        image_id = str(uuid.uuid4())
        
        # Return additional metadata for history tracking
        return jsonify({
            "image_data": data_uri, 
            "id": image_id,
            "timestamp": None,  # Will be set by JavaScript
            "filename": file.filename or "uploaded_image.png"
        })

    except Exception as e:
        print_warn(f"Error processing image: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 4971))
    app.run(debug=True, port=port, host="0.0.0.0")