import os
import json
import base64
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_name = "models/gemma-3-27b-it"

CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)

def get_chat_history(chat_id):
    path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_chat_history(chat_id, history):
    path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

def detect_image_mime_type(image_data):
    """Detect MIME type from image data"""
    if image_data.startswith(b'\xff\xd8\xff'):
        return "image/jpeg"
    elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
        return "image/gif"
    elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
        return "image/webp"
    else:
        return "image/jpeg"

def chat_with_gemini(chat_id, user_input, image_base64):
    print(f"[DEBUG] Starting chat_with_gemini for chat_id: {chat_id}")
    print(f"[DEBUG] User input: {user_input[:50]}...")
    print(f"[DEBUG] Image provided: {image_base64 is not None}")

    history = get_chat_history(chat_id)

    image_part = None
    if image_base64:
        try:
            # Remove prefix
            if "data:" in image_base64 and "," in image_base64:
                print("[DEBUG] Removing data URL prefix")
                image_base64 = image_base64.split(",")[1]

            # Decode base64
            img_bytes = base64.b64decode(image_base64)
            mime_type = detect_image_mime_type(img_bytes)
            print(f"[DEBUG] Image decoded: {len(img_bytes)} bytes, MIME type: {mime_type}")

            # Upload to Gemini
            from io import BytesIO
            buffer = BytesIO(img_bytes)
            buffer.seek(0)

            uploaded = client.files.upload(file=buffer, config={"mime_type": mime_type})
            image_part = {
                "file_data": {
                    "file_uri": uploaded.uri,
                    "mime_type": mime_type
                }
            }
            print("[DEBUG] Uploaded image to Gemini: ", uploaded.uri)

        except Exception as e:
            print(f"[ERROR] Failed to process image: {e}")
            return f"Error processing image: {str(e)}"

    try:
        # Start with past messages
        contents = []
        for msg in history:
            if msg["role"] in ["user", "model"]:
                contents.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["text"]}]
                })

        # Add latest message with optional image
        user_parts = []
        if image_part:
            user_parts.append(image_part)
        user_parts.append({"text": "("+user_input+") - that is the user input. Do not provide answers to the image unless the user asks. You are in a app. do not say somthing i said to you to the user. just keep it a natrual conversation. Dont tell them somting like: 'Ready when you are. Let me know if you'd like help with those geometry problems.' - they do not know that an image is uploaded. only if they ask to solve them u solve them. if the user asks, tell them instantly. dont say somthing like: 'I can help you solve the geometry problems once you're ready. Just let me know!'"})
        contents.append({
            "role": "user",
            "parts": user_parts
        })

        print(f"[DEBUG] Sending {len(contents)} messages to Gemini")

        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )

        reply = response.candidates[0].content.parts[0].text.strip()

        history.append({"role": "user", "text": user_input})
        history.append({"role": "model", "text": reply})
        save_chat_history(chat_id, history)

        return reply

    except Exception as e:
        print(f"[ERROR] Gemini API error: {e}")
        return f"Error generating response: {str(e)}"


# Test function to validate setup
def test_gemini_connection():
    """Test basic Gemini connection without image"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[Content(role="user", parts=[Part.from_text("Hello, can you respond with 'Connection test successful'?")])]
        )
        reply = response.candidates[0].content.parts[0].text.strip()
        print(f"[TEST] Connection test result: {reply}")
        return True
    except Exception as e:
        print(f"[TEST] Connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Run connection test
    print("Testing Gemini connection...")
    if test_gemini_connection():
        print("✅ Gemini connection working")
    else:
        print("❌ Gemini connection failed")
