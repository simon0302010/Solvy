import cv2
import base64
import supervision as sv
from ultralytics import YOLO
import dataclasses
import sys
from time import time

model = YOLO("models/field_detect_3.pt")

def run_inference(image_path): #, output_image_path):
    start_time = time()
    image = cv2.imread(image_path)

    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    detections = detections.with_nms(threshold=0.5)

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
    print(f"local inference took {round(time() - start_time, 2)} seconds.")
    return bounding_boxes_dict, base64.b64decode(annotated_image_base64)

if __name__ == "__main__":
    inference_result, image_base64 = run_inference(sys.argv[1]) #, "image.png")
    for bounding_box in inference_result:
        print(inference_result[bounding_box])
    print(image_base64[:100]) 