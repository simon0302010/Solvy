import os
import sys
import cv2
import extra_data
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import time

load_dotenv()

# Define Variables
inference = "local" # set inference either to roboflow or local
try: inference = sys.argv[2]
except IndexError: pass
gemini_model = "gemini-2.5-flash-preview-05-20"
#gemini_model = "gemini-2.0-flash"
worksheet_file_path = sys.argv[1]
gemini_prompts = extra_data.prompts
gemini_tools = extra_data.tools

if inference == "roboflow":
    print("using roboflow inference")
    from roboflow_inference import run_inference
elif inference == "local":
    print("using local inference")
    from local_inference import run_inference
else:
    print('please set inference to either "roboflow" or "local"')

# Initialize Gemini Client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

worksheet_file = Image.open(worksheet_file_path)
new_worksheet_file_path = os.path.splitext(worksheet_file_path)[0] + ".png"
if not worksheet_file_path == new_worksheet_file_path:
    print("input worksheet is not in png format, converting...")
    os.remove(worksheet_file_path)
    worksheet_file.save(new_worksheet_file_path)

# Function to add bounding boxes to any image
def add_bounding_boxes(bounding_box_data, image_path, output_filename=None):
    image = cv2.imread(image_path)

    for idx, bounding_box in enumerate(bounding_box_data):
        start_point = int(bounding_box["xmin"]), int(bounding_box["ymin"])
        end_point = int(bounding_box["xmax"]), int(bounding_box["ymax"])
        #middle_point = int(bounding_box["xmax"]) + 3, int((int(bounding_box["ymin"]) + int(bounding_box["ymax"])) / 2 + 5)
        middle_point = int((int(bounding_box["xmin"]) + int(bounding_box["xmax"])) / 2) - 7, int((int(bounding_box["ymin"]) + int(bounding_box["ymax"])) / 2 + 5)
        cv2.rectangle(image, start_point, end_point, color=(83,0,135), thickness=2)

        cv2.putText(
            image,
            str(idx + 1),
            middle_point,
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.5,
            color = (83,0,135),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    if output_filename is not None:
        cv2.imwrite(output_filename, image)
    return image

def run_gemini():
    inference_result, image_base64 = run_inference(new_worksheet_file_path) #, temp_worksheet_file_path)

    gemini_contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=image_base64
                ),
                types.Part.from_text(text=(gemini_prompts[0] + str(inference_result))),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        tools=gemini_tools,
        response_mime_type="text/plain",
        thinking_config=types.ThinkingConfig(thinking_budget=24576, include_thoughts=True)
    )

    while True:
        try:
            start_time = time.time()
            gemini_response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=gemini_contents,
                config=generate_content_config,
            )
            break
        except Exception as e:
            if "500" in str(e):
                print("Internal server error, retrying...")
                time.sleep(2)
            else:
                raise
                

    print(f"gemini took {round(time.time() - start_time, 2)} seconds.")
    
    print("Thoughts tokens:", gemini_response.usage_metadata.thoughts_token_count)
    print("Output tokens:", gemini_response.usage_metadata.candidates_token_count)

    function_call = {}
    for part in gemini_response.candidates[0].content.parts:
        if part.thought:
            print("Thought summary:")
            print(part.text)
            print()
        try:
            for key, value in part.function_call.args.items():
                function_call[key[9:]] = value
        except AttributeError:
            pass
    function_call = function_call[""]
    annotated_image = add_bounding_boxes(function_call, new_worksheet_file_path) #, "solved_worksheet.png")
    cv2.imshow("annotated image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run_gemini()

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