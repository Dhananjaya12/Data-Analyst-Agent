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
        
        prompt = f"""You are a senior data analyst. Generate ONLY fact-based insights.

# Question:
{state.user_query}

# Data Returned:
{state.data_retrieved}

# CRITICAL RULES:
1. Every insight MUST be directly verifiable from the data above. NO speculation.
2. NEVER produce filler or meta-insights like "No further breakdown available",
   "Limited data", "More data would be needed", or "N/A". If there's only one
   meaningful thing to say, say ONE insight and stop.
3. Quality > quantity. Fewer strong insights beat padded lists.

# Step 1: Classify the data shape FIRST:

## Case A: Single value (one number or single row with one key metric)
- Produce EXACTLY ONE insight that states the value plainly.
- Do NOT add a second insight. Do NOT say "no further breakdown."
- Example: Data = [{{"value": 20}}] → ONE insight: "The total number of customers is 20."

## Case B: Multiple rows with numeric values
- Identify highest, lowest, ratios BETWEEN actual values shown.
- Calculate percentages ONLY from numbers actually in the data.
- Produce 2-3 strong insights.

## Case C: Grouped/aggregated data
- Describe distribution (top N, concentration).
- Compare groups using actual numbers shown.
- Produce 2-4 strong insights.

## Case D: Empty result
- Produce ONE insight: "No data matched the query criteria."

# Examples:

Data: [{{"value": 20}}]
✅ GOOD (1 insight):
{{"data_shape": "single_value", "insights": ["The total number of customers is 20."]}}
❌ BAD (padded):
{{"data_shape": "single_value", "insights": ["The total number of customers is 20.", "No further breakdown available from this data"]}}

Data: [{{"total_revenue": 101350}}]
✅ GOOD (1 insight):
{{"data_shape": "single_value", "insights": ["Total revenue is $101,350."]}}
❌ BAD: "Revenue shows strong growth" (no time data!)
❌ BAD: "Performance is healthy" (no benchmark!)

Data: [{{"product": "Phone", "revenue": 72000}}, {{"product": "Laptop", "revenue": 36000}}]
✅ GOOD (2 insights):
{{"data_shape": "multi_row", "insights": [
  "Phone revenue ($72,000) is exactly 2x Laptop's ($36,000).",
  "Phone leads the two products with a $36,000 gap."
]}}

# Output (STRICT JSON only, no other text):
{{
  "data_shape": "single_value" | "multi_row" | "grouped" | "empty",
  "insights": [
    "Fact 1 directly from the data"
  ]
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