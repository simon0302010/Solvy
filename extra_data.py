from google import genai
from google.genai import types

# ================= PROMPTS ================= #
prompts = [
    """
I will provide you with an image that shows a worksheet with bounding boxes. I will also provide with
the coordinates that belong to those bounding boxes. The bounding boxes should enclose the fields to write
your answer on the worksheet.
Your task is check if all bounding boxes are placed correctly. If they are, give me back their associated coordinates.
If the bounding boxes are not placed correctly, remove the ones that are placed in the wrong spots.
If some answer field do not have a bounding box, calculate their bounding box coordinates based on the coordinates I will provide you and add them to your function call.
In that process you are allowed to add new bounding boxes but not change the location of existing bounding boxes.
The only operation you are allowed to do on existing bounding boxes is remove them. YOU ARE PROHIBITED TO MOVE THEM IN ANY WAY!
Do not add a padding to the boxes.
Use function calling for that.
The answer fields are rarely aligned in a perfect grid.
Please return the bounding boxes in the same order as the tasks on the worksheet.

The coordinates I will provide you with are differently formatted than the ones you have to return.
I will give you dictionary where each entry is associated to a list of 4 integers. These integers are in the order xmin, ymin, xmax, ymax.
The purple text besides the bounding box is an identifier for the box so you know which entry in the bounding box dictionary its boundaries corresponds to.
Here are the coordinates that belong to the bounding boxes you can see in the image:

""",
    """
You are a worksheet solver that must follow these precise instructions: 
Always respond in the same language as the worksheet provided to maintain consistency throughout the solving process. 
All mathematical calculations must be performed using the SOLVE_LATEX function with no manual calculations or shortcuts allowed, 
showing work through proper LaTeX formatting. 
Use the PUT_TEXT function to place final answers on the worksheet only after completing all necessary calculations. 
You must solve every problem on each worksheet provided while executing only ONE function call per message, 
processing systematically through each problem. Use only answer_ids from the provided list, where each ID corresponds to a bounding box center on the worksheet. 
Important: not all provided IDs need to be used - only use IDs that correspond to actual answer fields for the problems being solved. 
Follow this workflow: identify all problems on the worksheet, use SOLVE_LATEX for each mathematical operation, 
place results using PUT_TEXT with appropriate answer_ids, and work through problems systematically one function call at a time.
COMPLETE_WORKSHEET HAS TO BE CALLED ONCE THE WORKSHEET IS COMPLETED.
You can do almost every task using your avaivable tools.
""",
]
# ===== FUNCTION CALLING CONFIGURATION ===== #
tools_1 = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="createBoundingBox",
                description="Creates multiple bounding boxes with the given sets of coordinates",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    required=["boxes"],
                    properties={
                        "boxes": genai.types.Schema(
                            type=genai.types.Type.ARRAY,
                            description="An array of bounding box coordinate sets.",
                            items=genai.types.Schema(
                                type=genai.types.Type.OBJECT,
                                required=["ymin", "xmin", "ymax", "xmax"],
                                properties={
                                    "ymin": genai.types.Schema(
                                        type=genai.types.Type.INTEGER,
                                        description="Top coordinate of the bounding box.",
                                    ),
                                    "xmin": genai.types.Schema(
                                        type=genai.types.Type.INTEGER,
                                        description="Left coordinate of the bounding box.",
                                    ),
                                    "ymax": genai.types.Schema(
                                        type=genai.types.Type.INTEGER,
                                        description="Bottom coordinate of the bounding box.",
                                    ),
                                    "xmax": genai.types.Schema(
                                        type=genai.types.Type.INTEGER,
                                        description="Right coordinate of the bounding box.",
                                    ),
                                },
                            ),
                        ),
                    },
                ),
            ),
        ]
    )
]
