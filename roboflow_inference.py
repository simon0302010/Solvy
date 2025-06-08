from inference_sdk import InferenceHTTPClient
from time import time
from yaspin import yaspin
from logger import *
import supervision as sv
import numpy as np
import base64
import cv2
import sys
import os

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

def run_inference(image_bytes): #, output_image_path):
    image_bytes_new = image_bytes

    image_array = np.frombuffer(image_bytes_new, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    _, jpeg_bytes = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    image_base64_new = base64.b64encode(jpeg_bytes).decode('utf-8')

    with yaspin(text="Detecting bounding boxes", color="green") as sp:
        result = client.run_workflow(
            workspace_name="simon0302010",
            workflow_id="custom-workflow",
            images={
                "image": image_base64_new
            },
            use_cache=True # cache workflow definition for 15 minutes
        )
        #sp.write("> " + str(len(result[0]["predictions"]["predictions"])) + " bounding boxes found.")
        if len(result[0]["predictions"]["predictions"]) > 0:
            sp.ok("[✔]")
        else:
            sp.fail("[✖]")
            raise Exception("No bounding boxes detected")

    bounding_box_list = []
    bounding_box_dict = {}
    bounding_box_array = []
    for idx, bounding_box in enumerate(result[0]["predictions"]["predictions"]):
        if bounding_box["confidence"] >= 0.1:
            new_bounding_box = {}
            new_bounding_box_list = []
            new_bounding_box["xmin"] = round(bounding_box["x"] - (bounding_box["width"] / 2))
            new_bounding_box["ymin"] = round(bounding_box["y"] - (bounding_box["height"] / 2))
            new_bounding_box["xmax"] = round(bounding_box["x"] + (bounding_box["height"] / 2))
            new_bounding_box["ymax"] = round(bounding_box["y"] + (bounding_box["height"] / 2))
            bounding_box_list.append(new_bounding_box)
            new_bounding_box_list.append(new_bounding_box["xmin"])
            new_bounding_box_list.append(new_bounding_box["ymin"])
            new_bounding_box_list.append(new_bounding_box["xmax"])
            new_bounding_box_list.append(new_bounding_box["ymax"])
            bounding_box_dict[str(idx + 1)] = new_bounding_box_list
    for bounding_box in bounding_box_dict:
        bounding_box_array.append(bounding_box_dict[bounding_box])
    bounding_box_array = np.array(bounding_box_array)

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_RIGHT, text_scale=0.4, text_padding=2)

    detections = sv.Detections(
        xyxy=bounding_box_array,
        class_id=np.zeros((len(bounding_box_array),), dtype=int)
    )

    annotated_image = bounding_box_annotator.annotate(
        scene=image.copy(), detections=detections
    )

    labels = []
    for i in range(len(detections)):
        labels.append(str(i + 1))

    if labels:
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

    #cv2.imwrite(output_image_path, annotated_image)
    _, buffer = cv2.imencode('.png', annotated_image)
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    return bounding_box_dict, base64.b64decode(annotated_image_base64)

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        image_bytes = f.read()

    inference_result, image_base64 = run_inference(image_bytes) #, "image.png")
    for bounding_box in inference_result:
        print(inference_result[bounding_box])
    
    with open("image.png", "wb") as f:
        f.write(image_base64)