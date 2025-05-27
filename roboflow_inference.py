from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from time import time
import sys
import os

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

def run_inference(image_path):
    start_time = time()
    result = client.run_workflow(
        workspace_name="simon0302010",
        workflow_id="custom-workflow",
        images={
            "image": image_path
        },
        use_cache=True # cache workflow definition for 15 minutes
    )

    print(f"roboflow inference took {round(time() - start_time, 2)} seconds.")
    bounding_box_list = []
    bounding_box_dict = {}
    for idx, bounding_box in enumerate(result[0]["predictions"]["predictions"]):
        if bounding_box["confidence"] > 0.2:
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
    return(bounding_box_list, bounding_box_dict)

if __name__ == "__main__":
    bounding_box_list, bounding_box_dict = run_inference(sys.argv[1])
    for bounding_box in bounding_box_dict:
        print(bounding_box_dict[bounding_box])