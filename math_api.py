import requests
import ast
import json

latex_expression = "x = 2 + 5"

headers = {"content-type": "application/json"}
raw_data = {"LatexExpression": latex_expression,
            "clientInfo": {"mkt": "en"}}

answer = requests.post(url="https://mathsolver.microsoft.com/cameraexp/api/v1/solvelatex", headers=headers, json=raw_data)

if answer.status_code == 200:
    answer = eval(answer.text)["results"][0]["tags"][0]["actions"][0]["customData"]
    print(answer)
else:
    print(f"Could not get solution: {answer.status_code}")