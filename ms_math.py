import requests
import json

def solve_latex(latex_expression, language="en"):
    headers = {
        "content-type": "application/json"
    }
    raw_data = {
        "LatexExpression": latex_expression,
        "clientInfo": {"mkt": language}
    }

    answer = requests.post(
        url="https://mathsolver.microsoft.com/cameraexp/api/v1/solvelatex",
        headers=headers,
        json=raw_data
    )

    if answer.status_code != 200:
        raise(f"Failed to query Microsoft Math API: {answer.status_code}")

    solved_latex = json.loads(answer.text)["results"][0]["tags"][0]["actions"][0]["customData"]
    solved_latex = json.loads(solved_latex)["previewText"]
    if json.loads(solved_latex)["errorMessage"] != "": raise(json.loads(solved_latex))
    solved_latex = json.loads(solved_latex)["mathSolverResult"]
    if solved_latex["errorMessage"] != "": raise(json.loads(solved_latex))
    solved_latex = solved_latex["actions"]

    return solved_latex