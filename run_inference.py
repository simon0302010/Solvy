import cv2 # Make sure cv2 is imported
import supervision as sv
from ultralytics import YOLO


def run_inference(image_path, output_image_path, model_path="models/field_detect.pt"):
    model = YOLO(model_path)
    #image = cv2.imread(image_path)
    image = cv2.imread("test_worksheets/2.png")

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
    try:
        cv2.imwrite(output_image_path, annotated_image)
        print(f"Annotated image saved as {output_image_path} in the current directory.")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    run_inference("test_worksheets/2.png", "image.png")