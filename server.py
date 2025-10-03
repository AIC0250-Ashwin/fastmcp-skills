from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import json
import os

load_dotenv()

PORT = os.environ.get("PORT", 10000)

# ------------------------------------------------------------------
# Skills Store (hardcoded for now)
# ------------------------------------------------------------------
SKILLS = [
    {
        "name": "linear_regression_fit",
        "short_description": "Fit a linear regression model and evaluate its performance.",
        "use_cases": "Predicting one variable from others, establishing baseline models, checking linear relationships.",
        "long_description": "This skill takes a dataset, a target variable, and one or more predictor variables, fits a linear regression model, and returns coefficients, intercept, R², and MSE.",
        "type": "analysis",
        "language": "python",
        "code_snippet": """from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def linear_regression_fit(df: pd.DataFrame, target: str, features: list[str]):
    X = df[features]
    y = df[target]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return {
        "coefficients": dict(zip(features, model.coef_)),
        "intercept": model.intercept_,
        "r2": r2_score(y, y_pred),
        "mse": mean_squared_error(y, y_pred)
    }"""
    },
    {
        "name": "correlation_heatmap",
        "short_description": "Generate an interactive heatmap of correlations between numerical variables.",
        "use_cases": "Exploratory data analysis, identifying correlations, redundancy, or multicollinearity before modeling.",
        "long_description": "This skill computes pairwise correlations among numeric columns in a dataset and visualizes them as an interactive heatmap.",
        "type": "visualization",
        "language": "python",
        "code_snippet": """import pandas as pd
import plotly.express as px

def correlation_heatmap(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation Heatmap"
    )
    return fig"""
    }
]

# ------------------------------------------------------------------
# MCP server
# ------------------------------------------------------------------
mcp = FastMCP("skills-demo", host="0.0.0.0", port=PORT)


@mcp.tool()
def list_skills() -> str:
    """
    Return a JSON list of all available skills with metadata
    (name, short_description, use_cases, type, language).
    """
    results = []
    for s in SKILLS:
        results.append({
            "name": s["name"],
            "short_description": s["short_description"],
            "use_cases": s["use_cases"],
            "type": s["type"],
            "language": s["language"]
        })
    return json.dumps(results, indent=2)


@mcp.tool()
def get_skill(skill_name: str) -> str:
    """
    Return the code snippet for a skill, with its long description
    injected as comments.
    """
    skill = next((s for s in SKILLS if s["name"] == skill_name), None)
    if not skill:
        return f"# Unknown skill: {skill_name}"

    code = skill["code_snippet"]
    long_desc = skill.get("long_description", "")
    if long_desc:
        comment_block = "\n".join(
            "# " + line for line in long_desc.splitlines())
        code = f"{comment_block}\n\n{code}"
    return code


# ------------------------------------------------------------------
# Run the server
# ------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
