import cv2
import base64
import supervision as sv
from ultralytics import YOLO
from yaspin import yaspin
import dataclasses
import sys
import numpy as np
from logger import *
from time import time

model = YOLO("models/field_detect_3.pt")

def run_inference(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    with yaspin(text="Detecting bounding boxes", color="green") as sp:
        results = model(image)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections.with_nms(threshold=0.5)
        if len(detections) > 0:
            sp.ok("[✔]")
        else:
            sp.fail("[✖]")
            exit()


    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_RIGHT, text_scale=0.4, text_padding=2)

    annotated_image = image.copy()

    annotated_image = bounding_box_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    labels = []
    for i in range(len(detections)):
        labels.append(str(i + 1))
    #elif len(detections) > 0:
    #    labels = [
    #        f"cls:{class_id} {confidence:.2f}"
    #        for _, _, confidence, class_id, _, _
    #        in detections
    #    ]

    if labels:
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
    #cv2.imwrite(output_image_path, annotated_image)
    _, buffer = cv2.imencode('.png', annotated_image)
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    bounding_boxes = dataclasses.fields(detections)[0]
    bounding_boxes = getattr(detections, bounding_boxes.name)
    bounding_boxes_dict = {}
    for idx, bounding_box in enumerate(bounding_boxes):
        bounding_boxes_temp = []
        for temp_coordinate in bounding_box.tolist():
            bounding_boxes_temp.append(round(int(temp_coordinate)))
        bounding_boxes_dict[str(idx + 1)] = (bounding_boxes_temp)
    return bounding_boxes_dict, base64.b64decode(annotated_image_base64)

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        image_bytes = f.read()

    inference_result, image_base64 = run_inference(image_bytes) #, "image.png")
    for bounding_box in inference_result:
        print(inference_result[bounding_box])
    
    with open("image.png", "wb") as f:
        f.write(image_base64)