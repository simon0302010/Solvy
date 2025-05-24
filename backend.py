import os
import cv2
import extra_data
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from run_inference import run_inference

load_dotenv()

# Define Variables
gemini_model = "gemini-2.0-flash"
worksheet_file_path = "test_worksheets/2.png"
temp_worksheet_file_path = "temp/worksheet.png"
gemini_prompts = extra_data.prompts
gemini_tools = extra_data.tools

# Initialize Gemini Client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY2"))

worksheet_file = Image.open(worksheet_file_path)
worksheet_file.save(temp_worksheet_file_path)

# Function to add bounding boxes to any image
def add_bounding_boxes(bounding_box_data, image_path, output_filename):
    image = cv2.imread(image_path)

    for idx, bounding_box in enumerate(bounding_box_data):
        start_point = int(bounding_box["xmin"]), int(bounding_box["ymin"])
        end_point = int(bounding_box["xmax"]), int(bounding_box["ymax"])
        middle_point = int((int(bounding_box["xmin"]) + int(bounding_box["xmax"])) / 2), int((int(bounding_box["ymin"]) + int(bounding_box["ymax"])) / 2 + 5)
        cv2.rectangle(image, start_point, end_point, color=(0,0,255), thickness=1)

        cv2.putText(
            image,
            str(idx),
            middle_point,
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.5,
            color = (0,0,255),
            thickness=2
        )

    cv2.imwrite(output_filename, image)

def run_gemini():
    inference_result = run_inference(temp_worksheet_file_path, temp_worksheet_file_path)
    with open(temp_worksheet_file_path, "rb") as f:
        worksheet_bytes = f.read()

    gemini_contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=worksheet_bytes
                ),
                types.Part.from_text(text=(gemini_prompts[2] + str(inference_result))),
                #types.Part.from_text(text="Create these bounding boxes: " + str(inference_result)),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        tools=gemini_tools,
        response_mime_type="text/plain",
    )

    gemini_response = gemini_client.models.generate_content(
        model=gemini_model,
        contents=gemini_contents,
        config=generate_content_config,
    )
    
    print(gemini_response)

    function_call = {}
    for key, value in gemini_response.candidates[0].content.parts[0].function_call.args.items():
        function_call[key[9:]] = value
    function_call = function_call[""]
    print(function_call)
    add_bounding_boxes(function_call, worksheet_file_path, "temp/worksheet2.png")

# bounding_boxes = run_inference(temp_worksheet_file_path, temp_worksheet_file_path)

# add_bounding_boxes(function_call, worksheet_file_path, temp_worksheet_file_path)

run_gemini()

#with open(temp_worksheet_file_path, "rb") as f:
#    worksheet_bytes_modified = f.read()

#contents = [
#    types.Content(role="user", parts=[types.Part.from_bytes(mime_type="image/png", data=worksheet_bytes), types.Part.from_text(text=gemini_prompts[0])]),
#    types.Content(role="model", parts=[types.Part.from_text(text=("This was a function call: " + str(function_call)))]),
#    types.Content(role="user", parts=[types.Part.from_bytes(mime_type="image/png", data=worksheet_bytes_modified), types.Part.from_text(text=gemini_prompts[1])]),
#]

#start_time = time.time()
#gemini_response = gemini_client.models.generate_content(model=gemini_model, contents=contents, config=generate_content_config)
#print(f"second api call took {time.time() - start_time} seconds.")
#function_call = {}
#for key, value in gemini_response.candidates[0].content.parts[0].function_call.args.items():
#    function_call[key[9:]] = value
#function_call = function_call[""]
#add_bounding_boxes(function_call, worksheet_file_path, "temp/worksheet2.png")