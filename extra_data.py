from google import genai
from google.genai import types

# ================= PROMPTS ================= #
prompts = [
"""Give me a python list that contains a dictionary for each empty field on the provided worksheet.
The dictionary should be built like this (The values used are just placeholders): {"task_hierarchy": [5, 1], "answer_bounding_box": [842, 219, 1682, 476]}.
In this example the task is the first subtask of task 5. If the task hierarchy is larger the list can also be longer.
answer_bounding_box has to return a bounding box of the empty field to later input the result of the question.
The bouding box must be in the format [x1, y1, x2, y2] to be correctly displayed.
The bounding boxes have to be on the provided blank spots on the worksheets. These spots are usually indicated by underscores or empty rectangles
x = 0 and y = 0 are in the top left corner of the provided image.
Don't write an explaination of some kind, just the raw data.
""",
"""This is a image of the bounding boxes you set the first time. Please refine these positions and give me the EXACT same output you gave the last
time just with the refined coordinates of the bounding boxes. Don't write an explaination of some kind, just the raw data.
"""
]
# ================= FUNCTION CALLING ================= #
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="createBoundingBox",
                description="Creates multiple bounding boxes with the given sets of coordinates.",
                parameters=genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    required = ["boxes"],
                    properties = {
                        "boxes": genai.types.Schema(
                            type = genai.types.Type.ARRAY,
                            description = "An array of bounding box coordinate sets.",
                            items = genai.types.Schema(
                                type = genai.types.Type.OBJECT,
                                required = ["ymin", "xmin", "ymax", "xmax"],
                                properties = {
                                    "ymin": genai.types.Schema(
                                        type = genai.types.Type.INTEGER,
                                        description = "Top coordinate of the bounding box.",
                                    ),
                                    "xmin": genai.types.Schema(
                                        type = genai.types.Type.INTEGER,
                                        description = "Left coordinate of the bounding box.",
                                    ),
                                    "ymax": genai.types.Schema(
                                        type = genai.types.Type.INTEGER,
                                        description = "Bottom coordinate of the bounding box.",
                                    ),
                                    "xmax": genai.types.Schema(
                                        type = genai.types.Type.INTEGER,
                                        description = "Right coordinate of the bounding box.",
                                    ),
                                },
                            ),
                        ),
                    },
                ),
            ),
        ])
]