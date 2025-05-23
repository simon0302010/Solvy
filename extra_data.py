from google import genai
from google.genai import types

# ================= PROMPTS ================= #
prompts = [
"""
Identify all empty fields on the worksheet and provide their bounding boxes.
Return the bounding boxes in the order they appear from top to bottom, left to right on the worksheet.
""",
"""
This image shows the bounding boxes you initially provided, with each box numbered according to your original order.
Please review and refine the positions of these bounding boxes.
Return the updated bounding boxes in the same order.
To modify the bounding boxes, just use the same API that is used for creating them, just use the new values.
The data you first provided is NEVER perfect. You HAVE to refine it by calling the function call.
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