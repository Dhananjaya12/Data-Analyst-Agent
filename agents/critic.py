"""
Critic Agent - Validates analysis (multi-dimensional scoring)
"""

import json
import re
from .base import BaseAgent, ExecutionState
from logger_config import logger


class CriticAgent(BaseAgent):
    """Validates the analysis across 4 dimensions"""
    
    def __init__(self, llm):
        super().__init__("Critic", llm)
    
    async def execute(self, state: ExecutionState) -> ExecutionState:
        logger.info(f"🔍 [CRITIC] Validating...")
        
        prompt = f"""You are a strict data validation expert. Score this analysis on 4 dimensions.

# Original Question:
{state.user_query}

# Code Executed:
{state.query_generated}

# Data Returned:
{state.data_retrieved}

# Insights Generated:
{state.insights}

# Scoring Instructions:
Score EACH dimension independently from 0.0 to 1.0. Use the FULL range.

## 1. Logic Score (did the code answer the question correctly?)
- 0.9+: Logic is correct and appropriate
- 0.7-0.89: Logic works but has minor issues
- 0.5-0.69: Logic is partially wrong
- Below 0.5: Logic is broken

## 2. Data Score (is the returned data valid?)
- 0.9+: Data looks reasonable and complete
- 0.7-0.89: Data is OK with minor concerns
- 0.5-0.69: Data has issues (empty, NaN, suspicious values)
- Below 0.5: Data is clearly broken

## 3. Insights Score (are insights grounded in the data?)
- 0.9+: Every claim directly traceable to data
- 0.7-0.89: Mostly grounded, minor over-reaching
- 0.5-0.69: Some speculation beyond data
- Below 0.5: Heavily hallucinated

## 4. Completeness Score (fully answers the question?)
- 0.9+: Completely answers the question
- 0.7-0.89: Answers the main question, missing nuance
- 0.5-0.69: Partially answers
- Below 0.5: Misses the point

# IMPORTANT:
- Use varied scores across the range (NOT just 0.78 or 0.96)
- Be honest: if something is mediocre, score it mediocre (like 0.62 or 0.74)
- Different dimensions CAN have different scores

# Output (strict JSON only):
{{
  "logic_score": 0.XX,
  "data_score": 0.XX,
  "insights_score": 0.XX,
  "completeness_score": 0.XX,
  "error_type": null | "wrong_logic" | "bad_data" | "hallucinated_insight" | "incomplete",
  "issue_summary": "specific issue in one sentence, or 'None' if all valid",
  "recommendation": "finalize" | "refine" | "escalate"
}}

# Decision rules for recommendation:
- "finalize": All scores >= 0.8
- "refine": Any score between 0.5 and 0.79 (can be fixed)
- "escalate": Any score below 0.5 (fundamental problem)

Output:"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON safely
        report = None
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                report = json.loads(match.group())
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"⚠️ Failed to parse critic output: {e}")

        # Fallback if parsing failed
        if not report:
            logger.warning("⚠️ Using fallback critic report")
            report = {
                "logic_score": 0.5,
                "data_score": 0.5,
                "insights_score": 0.5,
                "completeness_score": 0.5,
                "error_type": "bad_data",
                "issue_summary": "Critic output was unparseable",
                "recommendation": "escalate"
            }

        # Compute overall confidence = MINIMUM of scores (weakest link)
        scores = [
            report.get('logic_score', 0.5),
            report.get('data_score', 0.5),
            report.get('insights_score', 0.5),
            report.get('completeness_score', 0.5)
        ]
        report['overall_confidence'] = round(min(scores), 2)

        # Derive boolean flags from scores (for backward compat)
        report['logic_valid'] = scores[0] >= 0.7
        report['data_valid'] = scores[1] >= 0.7
        report['insights_valid'] = scores[2] >= 0.7
        report['completeness_valid'] = scores[3] >= 0.7

        # Update state
        state.critic_report = report
        state.is_valid = all(s >= 0.7 for s in scores)
        state.confidence = report['overall_confidence']
        state.validation_feedback = report.get('issue_summary', 'No feedback')

        # Logging
        status = "Valid" if state.is_valid else "Invalid"
        recommendation = report.get('recommendation', 'unknown')
        logger.info(f"✅ {status} | Confidence: {state.confidence:.0%} | Recommendation: {recommendation}")
        logger.info(f"   Scores: logic={scores[0]:.2f}, data={scores[1]:.2f}, insights={scores[2]:.2f}, complete={scores[3]:.2f}")
        logger.info(f"   Issue: {state.validation_feedback}")

        return state