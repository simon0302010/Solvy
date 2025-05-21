# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=base64.b64decode(
                        """BINARY IMAGE"""
                    ),
                ),
                types.Part.from_text(text="""give me a list of all empty answer fields"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Here are the bounding box detections for the empty answer fields:"""),
                types.Part.from_function_call(
                    name="""createBoundingBox""",
                    args={"boxes":[{"xmax":443,"xmin":258,"ymax":278,"ymin":253},{"ymax":420,"ymin":395,"xmax":443,"xmin":258},{"ymax":559,"xmin":258,"xmax":443,"ymin":534}]},
                ),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name="""createBoundingBox""",
                    response={
                      "output": """""",
                    },
                ),
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
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
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text if chunk.function_calls is None else chunk.function_calls[0])

if __name__ == "__main__":
    generate()
