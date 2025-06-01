import os
import cv2
import ast
import time
import base64
import ms_math
import extra_data
from logger import *
from yaspin import yaspin
from google import genai
from google.genai import types

prompt = extra_data.prompts[1]

def answer_questions(image_opencv, possible_ids, model="gemini-2.0-flash"):
    _, buffer = cv2.imencode('.png', image_opencv)
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
    image_bytes = base64.b64decode(annotated_image_base64)

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    current_api_key = os.environ.get("GEMINI_API_KEY")
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=image_bytes,
                ),
                types.Part.from_text(text="Possible IDs: " + str(possible_ids))
            ],
        ),
    ]
    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="solve_latex",
                    description="Solves mathematical expressions and equations of any complexity, including basic arithmetic, algebra, calculus, statistics, and advanced mathematics. Returns the solution in valid LaTeX format. This function must be used for all mathematical calculations to ensure accuracy and proper formatting.",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["latex"],
                        properties={
                            "latex": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="The mathematical expression or equation to solve, written in LaTeX format",
                            ),
                        },
                    ),
                ),
                types.FunctionDeclaration(
                    name="put_text",
                    description="Inserts an answer into a specific question on the worksheet. Use this function to populate answers obtained from solve_latex. Each question should be answered exactly once after solving.",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["text", "question_id"],
                        properties={
                            "text": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="The answer text to insert, obtained from solve_latex function",
                            ),
                            "question_id": genai.types.Schema(
                                type=genai.types.Type.NUMBER,
                                description="The id of the field to put the answers in.",
                            ),
                        },
                    ),
                ),
                types.FunctionDeclaration(
                    name="complete_worksheet",
                    description="Marks the worksheet as complete after all questions have been answered using put_text. Call this function only when every question on the worksheet has been populated with a valid answer.",
                    parameters=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["total_questions"],
                        properties={
                            "total_questions": genai.types.Schema(
                                type=genai.types.Type.NUMBER,
                                description="Total number of questions that were answered on the worksheet",
                            ),
                            "summary": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                description="Brief summary of the worksheet completion status",
                            ),
                        },
                    ),
                ),
            ]
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
        candidate_count=1,
        system_instruction=[
            types.Part.from_text(text=prompt),
        ],
    )

    answers = {}

    while True:
        with yaspin(text="Solving task", color="green") as sp:
            while True:
                try:
                    gemini_response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    )
                    sp.ok("[✔]")
                    break
                except Exception as e:
                    if ("500" or "502" or "503") in str(e):
                        sp.write(bcolors.WARNING + "[!] " + bcolors.ENDC + "Internal Server Error, retrying...")
                        time.sleep(2)
                    elif "429" in str(e):
                        if os.environ.get("GEMINI_API_KEY_LIST"):
                            sp.write(bcolors.WARNING + "[!] " + bcolors.ENDC + "Rate limited, switching API Key...")
                            api_key_list = ast.literal_eval(os.environ.get("GEMINI_API_KEY_LIST"))
                            try:
                                current_index = api_key_list.index(current_api_key)
                            except ValueError:
                                current_index = 0  # fallback if current_api_key not in list
                            next_index = (current_index + 1) % len(api_key_list)
                            client = genai.Client(api_key=api_key_list[next_index])
                            current_api_key = api_key_list[next_index]
                        else:
                            sp.write(bcolors.WARNING + "[!] " + bcolors.ENDC + "Rate limited, waiting for 30 seconds...")
                            time.sleep(30)
                    else:
                        raise


        function_call = {}
        function_call_name = None
        text_output = None
        
        for part in gemini_response.candidates[0].content.parts:
            if part.text:
                text_output = str(part.text.strip())
            if part.function_call:
                try:
                    for key, value in part.function_call.args.items():
                        function_call[key] = value
                    function_call_name = str(part.function_call.name.strip())
                except AttributeError:
                    pass

        if text_output is not None:
            print_info("Text Output: " + text_output)
        if function_call_name is not None:
            print_info("Function Call: " + function_call_name + " = " + str(function_call))

        if function_call_name == "complete_worksheet":
            print_success("Worksheet Completed")
            return answers
        elif function_call_name == "solve_latex":
            math_response = ms_math.solve_latex(str(function_call["latex"]))
            func_response = ""
            func_response += ("Action: " + str(math_response["actionName"]) + "\n")
            func_response += ("Solution: " + str(math_response["solution"]).replace("$", ""))
            if "templateSteps" in math_response and math_response["templateSteps"]:
                func_response += ("\nSteps: " + str(math_response["templateSteps"][0]))
        elif function_call_name == "put_text":
            answers[str(function_call["question_id"])] = function_call["text"]
            func_response = "success"

        #print_info("Response from Function Call: " + func_response)
        #func_response = str(input("Enter function response: "))

        model_parts = []
        if text_output is not None:
            model_parts.append(types.Part.from_text(text=text_output))
        model_parts.append(
            types.Part.from_function_call(
                name=function_call_name,
                args=function_call
            )
        )
        contents.append(
            types.Content(
                role="model",
                parts=model_parts
            )
        )

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=function_call_name,
                        response={
                        "output": func_response
                        }
                    )
                ]
            )
        )