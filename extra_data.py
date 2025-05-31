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
This image shows the bounding boxes you initially provided, with each box numbered according to your original order.
Please review and refine the positions of these bounding boxes.
Return the updated bounding boxes in the same order.
To modify the bounding boxes, just use the same API that is used for creating them, just use the new values.
The data you first provided is NEVER perfect. You HAVE to refine it by calling the function call.
""",
"""
Create two bounding boxes 100x100 pixels big at 100,100 and 200,200 by giving the bounding box creation api the coordinates to such boxes.
"""
]
# ===== FUNCTION CALLING CONFIGURATION ===== #
tools_1 = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="createBoundingBox",
                description="Creates multiple bounding boxes with the given sets of coordinates",
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


tools_2 = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="solve_latex",
                description="Solves mathematical expressions and equations of any complexity, including basic arithmetic, algebra, calculus, statistics, and advanced mathematics. Returns the solution in valid LaTeX format. This function must be used for all mathematical calculations to ensure accuracy and proper formatting.",
                parameters=genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    required = ["latex"],
                    properties = {
                        "latex": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The mathematical expression or equation to solve, written in LaTeX format",
                        ),
                        "show_steps": genai.types.Schema(
                            type = genai.types.Type.BOOLEAN,
                            description = "Whether to include step-by-step solution process in the output",
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="put_text",
                description="Inserts an answer into a specific question on the worksheet. Use this function to populate answers obtained from solve_latex. Each question should be answered exactly once after solving.",
                parameters=genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    required = ["text", "question_number"],
                    properties = {
                        "text": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "The answer text to insert, obtained from solve_latex function",
                        ),
                        "question_number": genai.types.Schema(
                            type = genai.types.Type.NUMBER,
                            description = "The sequential number of the question being answered (starting from 1)",
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="complete_worksheet",
                description="Marks the worksheet as complete after all questions have been answered using put_text. Call this function only when every question on the worksheet has been populated with a valid answer.",
                parameters=genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    required = ["total_questions"],
                    properties = {
                        "total_questions": genai.types.Schema(
                            type = genai.types.Type.NUMBER,
                            description = "Total number of questions that were answered on the worksheet",
                        ),
                        "summary": genai.types.Schema(
                            type = genai.types.Type.STRING,
                            description = "Brief summary of the worksheet completion status",
                        ),
                    },
                ),
            ),
        ]
    )
]