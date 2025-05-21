import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ast import literal_eval
from PIL import Image
import cv2

bounding_box_function = {
    "name": "get_task_bounding_boxes",
    "description": "Returns task hierarchies and their bounding boxes.",
    "parameters": {
        "type": "object",
        "properties": {
            "input_data": {"type": "string", "description": "Input data to process"}
        },
        "required": ["input_data"]
    },
    "returns": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "task_hierarchy": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "answer_bounding_box": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["task_hierarchy", "answer_bounding_box"]
        }
    }
}


load_dotenv()

# Add bounding boxes to the emtpy fields on the worksheet
def add_bounding_boxes(bounding_box_data, image_path, output_filename):
    image = cv2.imread(image_path)

    for bounding_box in bounding_box_data:
        bounding_box_coordinates = bounding_box["answer_bounding_box"]
        start_point = int(bounding_box_coordinates[1]), int(bounding_box_coordinates[0])
        end_point = int(bounding_box_coordinates[3]), int(bounding_box_coordinates[2])
        cv2.rectangle(image, start_point, end_point, color=(0,255,0), thickness=1)

        cv2.putText(
            image,
            str(bounding_box["task_hierarchy"]),
            (int(bounding_box_coordinates[1]), int(bounding_box_coordinates[0]) - 10),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.5,
            color = (0, 255, 0),
            thickness=2
        )

    cv2.imwrite(output_filename, image)

with open("prompts.txt", "r") as prompts_file:
    prompts = literal_eval(prompts_file.read())

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = "gemini-2.0-flash"
worksheet_file_path = "test_worksheets/2.png"
worksheet_file = Image.open(worksheet_file_path)
worksheet_file.save("worksheet.png")
with open(worksheet_file_path, "rb") as f:
    worksheet_bytes = f.read()

# query gemini with the prompt and an image of the worksheet
gemini_response = gemini_client.models.generate_content(model=gemini_model, contents=[types.Part.from_bytes(mime_type="image/png", data=worksheet_bytes), prompts[0]]).text.replace("```", "")
print(gemini_response)

contents = [
    types.Content(role="user", parts=[types.Part.from_bytes(mime_type="image/png", data=worksheet_bytes), types.Part.from_text(text="""How are you doing?"""),],),
    types.Content(role="model", parts=[types.Part.from_text(text="""I'm doing great."""),],),
    types.Content(role="user", parts=[types.Part.from_text(text="""What was on the image?"""),],),
]

gemini_response = gemini_client.models.generate_content(model=gemini_model, contents=contents)
print(gemini_response.text)

# gemini_response = literal_eval(gemini_response.replace("python", ""))