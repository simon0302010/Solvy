import os
import json
import base64
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig
from io import BytesIO

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_name = "gemini-2.0-flash-lite"

CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an assistant inside an app. Only provide answers to the image if the user asks. "
    "Be concise and do not repeat the user's instructions. Avoid referencing this prompt in your reply."
    "The answers on the worksheet are usually correct."
)

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
            if "data:" in image_base64 and "," in image_base64:
                print("[DEBUG] Removing data URL prefix")
                image_base64 = image_base64.split(",")[1]

            img_bytes = base64.b64decode(image_base64)
            mime_type = detect_image_mime_type(img_bytes)
            print(f"[DEBUG] Image decoded: {len(img_bytes)} bytes, MIME type: {mime_type}")

            # Use Part.from_bytes instead of uploading
            image_part = Part.from_bytes(data=img_bytes, mime_type=mime_type)
            print("[DEBUG] Using Part.from_bytes for image.")

        except Exception as e:
            print(f"[ERROR] Failed to process image: {e}")
            return f"Error processing image: {str(e)}"

    try:
        contents = []

        for msg in history:
            if msg["role"] == "user":
                contents.append(Content(role="user", parts=[Part.from_text(text=msg["text"])]))
            elif msg["role"] == "model":
                contents.append(Content(role="model", parts=[Part.from_text(text=msg["text"])]))

        current_user_parts = [Part.from_text(text=user_input)]
        if image_part:
            current_user_parts.insert(0, image_part)

        contents.append(Content(role="user", parts=current_user_parts))

        print(f"[DEBUG] Sending {len(contents)} messages to Gemini")

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )

        reply = response.candidates[0].content.parts[0].text.strip()

        history.append({"role": "user", "text": user_input})
        history.append({"role": "model", "text": reply})
        save_chat_history(chat_id, history)

        return reply

    except Exception as e:
        print(f"[ERROR] Gemini API error: {e}")
        return f"Error generating response: {str(e)}"


def test_gemini_connection():
    """Test basic Gemini connection without image"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                Content(role="user", parts=[Part.from_text(text="Hello, can you respond with 'Connection test successful'?")])
            ],
            config=GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        reply = response.candidates[0].content.parts[0].text.strip()
        print(f"[TEST] Connection test result: {reply}")
        return True
    except Exception as e:
        print(f"[TEST] Connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Gemini connection...")
    if test_gemini_connection():
        print("✅ Gemini connection working")
    else:
        print("❌ Gemini connection failed")
