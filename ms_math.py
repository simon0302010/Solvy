import json

import requests
from yaspin import yaspin


def solve_latex(latex_expression, language="en"):
    with yaspin(text="Solving Latex equation", color="green") as sp:
        headers = {"content-type": "application/json"}
        raw_data = {
            "LatexExpression": latex_expression,
            "clientInfo": {"mkt": language},
        }

        answer = requests.post(
            url="https://mathsolver.microsoft.com/cameraexp/api/v1/solvelatex",
            headers=headers,
            json=raw_data,
        )

        if answer.status_code != 200:
            raise (f"Failed to query Microsoft Math API: {answer.status_code}")

        solved_latex = json.loads(answer.text)["results"][0]["tags"][0]["actions"][0][
            "customData"
        ]
        solved_latex = json.loads(solved_latex)["previewText"]
        if json.loads(solved_latex)["errorMessage"] != "":
            return json.loads(solved_latex)["errorMessage"]
        solved_latex = json.loads(solved_latex)["mathSolverResult"]
        if solved_latex["errorMessage"] != "":
            return solved_latex["errorMessage"]
        solved_latex = solved_latex["actions"]

        sp.ok("[✔]")
        return solved_latex[0]


if __name__ == "__main__":
    result = solve_latex("x^2 = 9")
    print("Action: " + str(result["actionName"]))
    print("Solution: " + str(result["solution"]))
    print("Steps: " + str(result["templateSteps"][0]))
