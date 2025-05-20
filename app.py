import os
from dotenv import load_dotenv
from google import genai
from ast import literal_eval
import cv2
import json

load_dotenv()

with open("prompts.txt", "r") as prompts_file:
    prompts = literal_eval(prompts_file.read())

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = "gemini-2.0-flash"
worksheet_file_path = "test_worksheets/2.png"
# load image for later use
image = cv2.imread(worksheet_file_path)
image_height, image_width, _ = image.shape
worksheet_file = gemini_client.files.upload(file=worksheet_file_path)

# add image dimensions to first prompt
first_prompt = prompts[0] + str(image_width) + "x" + str(image_height)

# query gemini with the prompt and an image of the worksheet
gemini_response = gemini_client.models.generate_content(model=gemini_model, contents=[worksheet_file, first_prompt]).text.replace("```", "")

gemini_response = json.loads(gemini_response.replace("python", ""))

print(f"response from gemini: {gemini_response}")
print(type(gemini_response))

# Add bounding boxes to the emtpy fields on the worksheet
for bounding_box in gemini_response:
    bounding_box = bounding_box["answer_bounding_box"]
    start_point = int(bounding_box[0]), int(bounding_box[1])
    end_point = int(bounding_box[2]), int(bounding_box[3])

    cv2.rectangle(image, start_point, end_point, color=(0,255,0), thickness=1)

    cv2.putText(
        image,
        #bounding_box["label"],
        "label",
        (int(bounding_box[0]), int(bounding_box[1]) - 10),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        color = (0, 255, 0),
        thickness=2
    )

cv2.imwrite("example_with_bounding_boxes.jpg", image)

print(prompts)