"""
Planner Agent - Decomposes queries into steps
"""

from .base import BaseAgent, ExecutionState
from logger_config import logger
from caching import cache_manager

class PlannerAgent(BaseAgent):
    """Breaks down complex queries into steps"""
    
    def __init__(self, llm, context=None):
        super().__init__("Planner", llm)
        self.context = context

    async def execute(self, state: ExecutionState) -> ExecutionState:
        logger.info(f"\n🧠 [PLANNER] {state.user_query}")
        # dataset_context = self.context.build_short() if self.context else ""
        conversation_context = state.conversation_context or ""
        dataset_context = state.selected_files_context or "No file context available"
#         prompt = f"""Break down this question into 2-3 brief steps.

# Question: {state.user_query}

# Steps:"""

        prompt = f"""You are a data analysis planner. Break queries into executable pandas steps.

# Available Datasets:
{dataset_context}

# Previous Context:
{conversation_context}

# Planning Rules:
1. Each step = ONE concrete pandas operation (filter, groupby, aggregate, sort, join, calculate)
2. Reference datasets by their exact IDs shown above
3. Use EXACT column names from the schema (never invent columns)
4. For multi-dataset queries, include explicit join step with the join key
5. Order steps logically: filter → join → group → aggregate → sort
6. If user refers to previous questions (e.g., "compare that"), use context above
7. Do NOT write code - only describe operations

# Quality Bar:
✅ GOOD: "Filter sales where revenue > 1000"
✅ GOOD: "Join sales and products on product_id"
✅ GOOD: "Group by category, aggregate sum of revenue"

❌ BAD: "Analyze the data" (too vague)
❌ BAD: "Clean the data" (not actionable)
❌ BAD: "Use the customers table" (table doesn't exist)

# User Query:
{state.user_query}

# Output (strict JSON, no other text):
{{
  "approach": "one-sentence summary of your strategy",
  "steps": [
    "Step 1 - concrete operation",
    "Step 2 - concrete operation",
    "Step 3 - concrete operation"
  ]
}}"""
        
        # CHECK LLM CACHE FIRST
        # cached_response = cache_manager.get_llm_cache(prompt)
        cached_response = cache_manager.get_llm_cache(
    agent="Planner",
    user_query=state.user_query,
    file_ids=state.selected_file_ids or [],
    prompt_preview=prompt[:200],
)
        
        if cached_response:
            print("✅ Using cached LLM response")
            state.plan = cached_response
        else:
            print("🔄 Calling LLM (no cache)")
            # Call LLM
            response = self.llm.invoke(prompt)
            plan = response.content if hasattr(response, 'content') else str(response)
            
            # SAVE TO CACHE
            # cache_manager.set_llm_cache(prompt, plan)
            cache_manager.set_llm_cache(
        agent="Planner",
        user_query=state.user_query,
        file_ids=state.selected_file_ids or [],
        response=plan,
        model=getattr(self.llm, "model", "") or getattr(self.llm, "model_name", ""),
    )
            state.plan = plan
        
        return state
