import base64
import os
import uuid
import cv2
from flask import Flask, jsonify, render_template, request
import backend
from logger import print_info, print_warn
from chat import chat_with_gemini, get_chat_history

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session-based features

@app.route("/")
def home():
    return render_template("home.html", questions=[])

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
    
    try:
        image_bytes = file.read()
        print_info(f"Received image: {len(image_bytes)} bytes")
        print_info("Processing image...")

        processed_image = backend.process_image(image_bytes)
        success, encoded_image = cv2.imencode(".png", processed_image)

        if not success:
            return jsonify({"error": "Failed to encode image"}), 500
        
        base64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_image}"
        image_id = str(uuid.uuid4())

        return jsonify({
            "image_data": data_uri,
            "id": image_id,
            "timestamp": None,
            "filename": file.filename or "uploaded_image.png"
        })

    except Exception as e:
        print_warn(f"Error processing image: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/chat")
def chat_root():
    return render_template("chat.html", chat_id=None)

@app.route("/chat/<chat_id>", methods=["GET"])
def chat_page(chat_id):
    history = get_chat_history(chat_id)
    return render_template("chat_uuid.html", chat_id=chat_id, history=history)

@app.route("/api/chat/<chat_id>", methods=["POST", "GET"])
def handle_chat(chat_id):
    if request.method == "POST":
        try:
            data = request.get_json()
            user_input = data.get("message", "").strip()
            image_base64 = data.get("image_base64")

            print(image_base64)
            if not user_input:
                return jsonify({"error": "Message cannot be empty"}), 400

            if not image_base64:
                print_warn("❌ NO IMAGE PROVIDED TO GEMINI — Check your frontend request.")
                return jsonify({"error": "Image is required but not provided"}), 400

            print_info("📨 Sending message + image to Gemini")
            print_info(f"✏️ Message: {user_input}")
            print_info(f"🖼️ Image size: {len(image_base64)} base64 chars")

            reply = chat_with_gemini(chat_id, user_input, image_base64)
            print_info(f"✅ Gemini reply (truncated): {reply[:100]}...")

            return jsonify({"reply": reply})

        except Exception as e:
            print_warn(f"🚨 Error in chat endpoint: {e}")
            return jsonify({"error": "Failed to process message"}), 500

    # GET
    history = get_chat_history(chat_id)
    return jsonify({"history": history})

@app.route("/settings")
def settings():
    return render_template("settings.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4971))
    app.run(debug=True, port=port, host="0.0.0.0")
