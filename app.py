import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ast import literal_eval
import cv2

load_dotenv()

with open("prompts.txt", "r") as prompts_file:
    prompts = literal_eval(prompts_file.read())

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = "gemini-2.0-flash"
worksheet_file_path = "test_worksheets/2.png"
worksheet_file = gemini_client.files.upload(file=worksheet_file_path)

# query gemini with the prompt and an image of the worksheet
gemini_response = gemini_client.models.generate_content(model=gemini_model, contents=[worksheet_file, prompts[0]]).text.replace("```", "")

print(f"response from Gemini: {gemini_response}")

gemini_response = literal_eval(gemini_response.replace("python", ""))

# Add bounding boxes to the emtpy fields on the worksheet
def add_bounding_boxes(bounding_box_data, image_path):
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

    cv2.imwrite("example_with_bounding_boxes.jpg", image)
    return 1

contents = [
    types.Content(role="user", parts=[types.Part.from_text(text="""hi"""),],),
    types.Content(role="model", parts=[types.Part.from_text(text="""Hi there! How can I help you today?"""),],),
    types.Content(role="user", parts=[types.Part.from_text(text="""INSERT_INPUT_HERE"""),],),
]