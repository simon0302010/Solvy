import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ast import literal_eval
from PIL import Image
import cv2
import extra_data

load_dotenv()

# Define Variables
gemini_model = "gemini-2.0-flash"
worksheet_file_path = "test_worksheets/2.png"
prompts = extra_data.prompts

# Initialize Gemini Client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Function to add bounding boxes to any image
def add_bounding_boxes(bounding_box_data, image_path, output_filename):
    image = cv2.imread(image_path)

    for idx, bounding_box in enumerate(bounding_box_data):
        bounding_box_coordinates = bounding_box["answer_bounding_box"]
        start_point = int(bounding_box_coordinates[1]), int(bounding_box_coordinates[0])
        end_point = int(bounding_box_coordinates[3]), int(bounding_box_coordinates[2])
        middle_x = (int(bounding_box_coordinates[1]) + int(bounding_box_coordinates[3])) / 2
        middle_y = (int(bounding_box_coordinates[0]) + int(bounding_box_coordinates[2])) / 2
        cv2.rectangle(image, start_point, end_point, color=(0,255,0), thickness=1)

        cv2.putText(
            image,
            idx
            (middle_x, middle_y),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.5,
            color = (0, 255, 0),
            thickness=2
        )

    cv2.imwrite(output_filename, image)

worksheet_file = Image.open(worksheet_file_path)
worksheet_file.save("temp/worksheet.png")
with open("temp/worksheet.png", "rb") as f:
    worksheet_bytes = f.read()

# query gemini with the prompt and an image of the worksheet
gemini_response = gemini_client.models.generate_content(model=gemini_model, contents=[types.Part.from_bytes(mime_type="image/png", data=worksheet_bytes), prompts[0]])
print(gemini_response.text)

contents = [
    types.Content(role="user", parts=[types.Part.from_bytes(mime_type="image/png", data=worksheet_bytes), types.Part.from_text(text="""How are you doing?"""),],),
    types.Content(role="model", parts=[types.Part.from_text(text="""I'm doing great."""),],),
    types.Content(role="user", parts=[types.Part.from_text(text="""What was on the image?"""),],),
]

gemini_response = gemini_client.models.generate_content(model=gemini_model, contents=contents)
print(gemini_response.text)