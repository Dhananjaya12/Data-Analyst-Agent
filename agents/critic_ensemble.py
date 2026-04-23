"""
Ensemble Critics - Multiple validators with different perspectives.
Each critic returns a dict of ONLY its own fields, so they can run in parallel
in LangGraph without race conditions.
"""

import json
import re
from .base import BaseAgent, ExecutionState
from logger_config import logger


def _parse_score(response_text: str, score_key: str, default: float = 0.5):
    """
    Robust JSON parser for critic responses. Handles:
    - markdown fences (```json ... ```)
    - Python None/True/False instead of null/true/false
    - trailing commas
    - prose before/after the JSON
    - out-of-range scores
    Returns (score, issue).
    """
    if not response_text:
        return default, "Empty response"

    text = response_text.strip()

    # Strip markdown fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Find the JSON object
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if not brace:
        return default, f"No JSON in response: {text[:80]}"
    candidate = brace.group()

    # Fix common LLM output quirks before parsing
    candidate = re.sub(r"\bNone\b", "null", candidate)
    candidate = re.sub(r"\bTrue\b", "true", candidate)
    candidate = re.sub(r"\bFalse\b", "false", candidate)
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)

    try:
        report = json.loads(candidate)
    except json.JSONDecodeError as e:
        # Regex fallback — pull the score directly
        m = re.search(rf'"{score_key}"\s*:\s*([\d.]+)', candidate)
        if m:
            try:
                return float(m.group(1)), "Partial parse"
            except ValueError:
                pass
        return default, f"Parse error: {str(e)[:60]}"

    score = report.get(score_key, default)
    issue = report.get("issue") or report.get("reason") or "OK"

    try:
        score = float(score)
        if not (0.0 <= score <= 1.0):
            return default, f"Score out of range: {score}"
    except (TypeError, ValueError):
        return default, "Non-numeric score"

    return score, issue


class LogicCritic(BaseAgent):
    def __init__(self, llm):
        super().__init__("LogicCritic", llm)

    async def execute(self, state: ExecutionState) -> dict:
        prompt = f"""You are a code logic validator. Evaluate ONLY the logic.

Question: {state.user_query}
Code: {state.query_generated}
Data returned: {state.data_retrieved}

Score the LOGIC from 0-1.0:
- 0.9+: Logic is correct and answers the question
- 0.7-0.89: Logic works but has minor issues
- 0.5-0.69: Logic is partially wrong
- Below 0.5: Logic is fundamentally broken

CRITICAL: Respond with ONLY a JSON object. No markdown, no prose.
Use JSON syntax: null (not None), true/false (not True/False).

Example valid output:
{{"logic_score": 0.95, "issue": null}}

Your output:"""

        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        score, issue = _parse_score(text, 'logic_score')
        return {"logic_score": score, "logic_issue": issue}


class DataCritic(BaseAgent):
    def __init__(self, llm):
        super().__init__("DataCritic", llm)

    async def execute(self, state: ExecutionState) -> dict:
        prompt = f"""You are a data quality validator. Evaluate ONLY the data.

Question: {state.user_query}
Data returned: {state.data_retrieved}

Score the DATA from 0-1.0:
- 0.9+: Data looks reasonable and complete
- 0.7-0.89: Data is OK with minor concerns
- 0.5-0.69: Data has issues (empty, NaN, suspicious values)
- Below 0.5: Data is clearly broken

CRITICAL: Respond with ONLY a JSON object. No markdown, no prose.
Use JSON syntax: null (not None), true/false (not True/False).

Example valid output:
{{"data_score": 0.95, "issue": null}}

Your output:"""

        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        score, issue = _parse_score(text, 'data_score')
        return {"data_score": score, "data_issue": issue}


class InsightsCritic(BaseAgent):
    def __init__(self, llm):
        super().__init__("InsightsCritic", llm)

    async def execute(self, state: ExecutionState) -> dict:
        prompt = f"""You are an insights validator. Evaluate ONLY the insights.

Question: {state.user_query}
Data: {state.data_retrieved}
Insights: {state.insights}

Score the INSIGHTS from 0-1.0:
- 0.9+: Every insight is grounded in data, no speculation
- 0.7-0.89: Mostly grounded, minor over-reaching
- 0.5-0.69: Some speculation beyond data
- Below 0.5: Heavily hallucinated

CRITICAL: Respond with ONLY a JSON object. No markdown, no prose.
Use JSON syntax: null (not None), true/false (not True/False).

Example valid output:
{{"insights_score": 0.95, "issue": null}}

Your output:"""

        response = self.llm.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)
        score, issue = _parse_score(text, 'insights_score')
        return {"insights_score": score, "insights_issue": issue}