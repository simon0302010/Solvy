import cv2
import supervision as sv
from ultralytics import YOLO
import dataclasses
from time import time

model = YOLO("models/field_detect_2.pt")

def run_inference(image_path, output_image_path):
    start_time = time()
    image = cv2.imread(image_path)

    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    detections = detections.with_nms(threshold=0.5)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

    annotated_image = image.copy()

    annotated_image = bounding_box_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    possible_labels = ['space']
    labels = []
    if len(detections) > 0:
        labels = [
            f"{possible_labels[class_id]} {confidence:.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]
    elif len(detections) > 0:
        labels = [
            f"cls:{class_id} {confidence:.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]

    if labels:
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
    cv2.imwrite(output_image_path, annotated_image)
    
    bounding_boxes = dataclasses.fields(detections)[0]
    bounding_boxes = getattr(detections, bounding_boxes.name)
    bounding_boxes_list = []
    for bounding_box in bounding_boxes:
        bounding_boxes_list.append(bounding_box.tolist())
    print(f"inference took {time() - start_time} seconds.")
    return bounding_boxes_list

if __name__ == "__main__":
    print(run_inference("test_worksheets/3.jpg", "image.png"))