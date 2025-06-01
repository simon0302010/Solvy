import io
import os
import time

import cv2
import easyocr
import statistics
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from yaspin import yaspin

from answer_questions import answer_questions
import click
import extra_data
from logger import *

load_dotenv()

# Define Variables
inference = "roboflow" # set inference either to roboflow or local
gemini_prompts = extra_data.prompts
gemini_tools = extra_data.tools_1

if inference == "roboflow":
    print_info("Using Roboflow inference")
    from roboflow_inference import run_inference
elif inference == "local":
    print_info("Using Local inference")
    from local_inference import run_inference
else:
    print_fail('Please set inference to either "roboflow" or "local"')
    exit()

# Initialize Gemini Client
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def list_to_dict(old_list):
    new_dict = {}
    for idx, entry in enumerate(old_list):
        new_dict[str(idx + 1)] = entry
    return new_dict

def prepare_image(image_bytes, max_dim=1024):
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)

    w, h = image.size
    scale = min(max_dim / h, max_dim / w, 1.0)
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()

def get_mean_font_size(image_bytes):
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image_bytes)
    heights = [max(p[1] for p in bbox) - min(p[1] for p in bbox) 
               for bbox, text, conf in results if conf > 0.5]
    return statistics.mean(heights) if heights else 0

# Function to add bounding boxes to any image
def add_bounding_boxes(bounding_box_data, image, output_filename=None):
    for idx, bounding_box in enumerate(bounding_box_data):
        start_point = int(bounding_box["xmin"]), int(bounding_box["ymin"])
        end_point = int(bounding_box["xmax"]), int(bounding_box["ymax"])
        middle_point = int((int(bounding_box["xmin"]) + int(bounding_box["xmax"])) / 2) - 7, int((int(bounding_box["ymin"]) + int(bounding_box["ymax"])) / 2 + 5)
        cv2.rectangle(image, start_point, end_point, color=(251,81,163), thickness=2)

        cv2.putText(
            image,
            str(idx + 1),
            middle_point,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            color=(0,75,0),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    if output_filename is not None:
        cv2.imwrite(output_filename, image)
    print_success(f"Added {str(len(bounding_box_data))} bounding boxes")
    return image

def add_text(text_dict, bounding_boxes_dict, font_size, image):
    for text_id in text_dict:
        answer_text = str(text_dict[text_id])
        answer_bounding_box = bounding_boxes_dict[text_id]
        text_start = (
            int(answer_bounding_box["xmin"]) + 2,
            int((int(answer_bounding_box["ymin"]) + int(answer_bounding_box["ymax"])) / 2 + 5)
        )
        text_start = (int(text_start[0]), int(text_start[1]))

        cv2.putText(
            image,
            answer_text,
            text_start,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=font_size/30,
            color=(0,0,0),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    return image

def process_image(image_bytes, gemini_model_1="gemini-2.5-flash-preview-05-20", gemini_model_2="gemini-2.0-flash"):
    image_bytes = prepare_image(image_bytes)
    mean_font_size = get_mean_font_size(image_bytes)
    print_info(f"Mean font size on image is {mean_font_size}")
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_opencv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    solved_worksheet = image_opencv.copy()

    print_info(f"Using {gemini_model_1} to fix the bounding boxes")
    print_info(f"Using {gemini_model_2} to solve the questions")
    inference_result, image_base64 = run_inference(image_bytes)

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
        candidate_count=1,
        tools=gemini_tools,
        response_mime_type="text/plain",
        thinking_config=types.ThinkingConfig(thinking_budget=24576, include_thoughts=True)
    )

    with yaspin(text="Fixing misplaced bounding boxes", color="green") as sp:
        while True:
            try:
                gemini_response = gemini_client.models.generate_content(
                    model=gemini_model_1,
                    contents=gemini_contents,
                    config=generate_content_config,
                )
                sp.ok("[✔]")
                break
            except Exception as e:
                if ("500" or "502" or "503") in str(e):
                    sp.write(bcolors.WARNING + "[!] " + bcolors.ENDC + "Internal Server Error, retrying...")
                    time.sleep(2)
                else:
                    raise
                    
    print_info("Thought tokens: " + str(gemini_response.usage_metadata.thoughts_token_count))
    print_info("Output tokens: " + str(gemini_response.usage_metadata.candidates_token_count))

    function_call = {}
    for part in gemini_response.candidates[0].content.parts:
        if part.thought:
            with yaspin(text="Generating thought summary", color="green") as sp:
                thought_summary = ("Thought summary: " + gemini_client.models.generate_content(model="gemma-3-12b-it", contents=f"make a detailed 50 word summary: {part.text}").text.strip())
                sp.ok("[✔]")
            print_info(thought_summary)
        try:
            for key, value in part.function_call.args.items():
                function_call[key] = value
        except AttributeError:
            pass
    
    bounding_boxes_dict = list_to_dict(function_call["boxes"])
    annotated_image = add_bounding_boxes(function_call["boxes"], image_opencv) #, "solved_worksheet.png")
    
    answers = answer_questions(annotated_image, list(bounding_boxes_dict.keys()), model=gemini_model_2)

    solved_worksheet = add_text(answers, bounding_boxes_dict, mean_font_size, solved_worksheet)

    if click.confirm(bcolors.ORANGE + "[?] " + bcolors.ENDC + "Do you want to view the solved worksheet?", default=True): cv2.imshow("Solved Worksheet", solved_worksheet); cv2.waitKey(0); cv2.destroyAllWindows()

    return solved_worksheet