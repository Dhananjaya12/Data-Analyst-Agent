"""
Insight Generator Agent - Fact-based insights with data shape awareness
"""

import json
import re
from .base import BaseAgent, ExecutionState
from logger_config import logger


class InsightGeneratorAgent(BaseAgent):
    """Extracts fact-based insights grounded in actual data"""
    
    def __init__(self, llm):
        super().__init__("InsightGenerator", llm)
    
    async def execute(self, state: ExecutionState) -> ExecutionState:
        logger.info(f"💡 [INSIGHTS] Analyzing...")
        
        # Handle empty/error state
        if not state.data_retrieved:
            state.insights = ["No data available to generate insights."]
            logger.info(f"✅ No data - skipping insights")
            return state
        
        prompt = f"""You are a data analyst. Answer the user's question using the data.

# Question:
{state.user_query}

# Data:
{state.data_retrieved}

# Code (for context):
{state.query_generated}

# Rules:
1. First insight = direct answer to the question, using specific values from the data.
2. Optional 1-2 follow-up insights ONLY if they add real context (comparisons, surprising patterns).
3. NEVER say: "no further breakdown", "limited data", "N/A", or similar filler.
4. NEVER invent values or speculate beyond the data.

# Examples:

Q: "How many customers?"
Data: [{{"value": 20}}]
→ {{"data_shape": "single_value", "insights": ["You have 20 customers."]}}

Q: "Which product has highest revenue?"
Data: [{{"product": "Phone", "revenue": 72000}}, {{"product": "Laptop", "revenue": 36000}}]
→ {{"data_shape": "grouped", "insights": [
    "Phone has the highest revenue at $72,000.",
    "That's 2× Laptop ($36,000), the runner-up."
  ]}}

# Output (JSON only):
{{
  "data_shape": "single_value" | "multi_row" | "grouped" | "empty",
  "insights": ["Direct answer", "Optional context"]
}}

Output:"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON safely
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                state.insights = parsed.get('insights', [])
                shape = parsed.get('data_shape', 'unknown')
                logger.info(f"✅ Insights generated (shape: {shape}, count: {len(state.insights)})")
            else:
                # Fallback - no JSON found
                logger.warning("⚠️ No JSON in response, using raw text")
                state.insights = [line.strip() for line in response_text.split('\n') 
                                  if line.strip() and not line.strip().startswith('```')]
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse insights: {e}")
            state.insights = [response_text[:200]]  # Truncated fallback
        
        return state