import os
from dotenv import load_dotenv
from google import genai
import json

load_dotenv()

with open("prompts.txt", "r") as prompts_file:
    prompts = eval(prompts_file.read())

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = "gemini-2.0-flash"
worksheet_file = "test_worksheets/2.png"
worksheet_file = gemini_client.files.upload(file=worksheet_file)

gemini_response = gemini_client.models.generate_content(model=gemini_model, contents=[worksheet_file, prompts[0]]).text.replace("```", "")

#gemini_response = eval(gemini_response)

print(gemini_response)