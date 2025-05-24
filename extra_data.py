from google import genai
from google.genai import types

# ================= PROMPTS ================= #
prompts = [
"""
You are analyzing a worksheet image containing multiple empty fields that users must fill out.

### Objective:
Identify all input fields and ensure each one is enclosed in a single, correct bounding box.

### Instructions:
- Some fields already have bounding boxes (with numeric IDs and pixel coordinates).
- If all fields are correctly represented, return them unchanged.
- If any bounding boxes are:
  - overlapping,
  - duplicated (represent the same field multiple times),
  - or missing,
  then correct or infer their coordinates.
- Use the spatial layout of the worksheet and existing boxes to guide your corrections.
- In all cases, return a **complete list** of bounding boxes, including existing, corrected, and inferred ones.
- The coordinates I will provide you with are in the format xmin, ymin, xmax, ymax.

Please analyze the image and return the final bounding boxes using function calling.

The bounding boxes on the image and the coordinates I will provide you with will ALWAYS match up.

These are the coordinates of the bounding boxes:


""",
"""
This image shows the bounding boxes you initially provided, with each box numbered according to your original order.
Please review and refine the positions of these bounding boxes.
Return the updated bounding boxes in the same order.
To modify the bounding boxes, just use the same API that is used for creating them, just use the new values.
The data you first provided is NEVER perfect. You HAVE to refine it by calling the function call.
""",
"""
Describe what you see in detail. Also do a test function call.
"""
]
# ===== FUNCTION CALLING CONFIGURATION ===== #
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