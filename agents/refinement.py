"""
Refinement Agent - Self-heals failed analyses based on critic feedback
"""

import pandas as pd
import re
import ast
from .base import BaseAgent, ExecutionState
from logger_config import logger


class RefinementAgent(BaseAgent):
    """Fixes failed analyses using critic feedback"""
    
    def __init__(self, llm, registry):
        super().__init__("Refinement", llm)
        self.registry = registry
    
    def _is_code_safe(self, code: str) -> bool:
        """Same safety check as DataAnalyst"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return False
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["eval", "exec", "open", "__import__"]:
                            return False
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id in ["os", "sys", "subprocess", "shutil"]:
                            return False
            return True
        except:
            return False
    
    def _clean_code(self, code: str) -> str:
        """Extract code from markdown fences"""
        code = code.strip()
        match = re.search(r"```(?:python)?\n?(.*?)```", code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        return code
    
    async def execute(self, state: ExecutionState) -> ExecutionState:
        logger.info(f"🔧 [REFINEMENT] Fixing based on critic feedback...")
        
        # Get critic feedback
        critic_report = getattr(state, 'critic_report', {})
        error_type = critic_report.get('error_type', 'unknown')
        issue_summary = critic_report.get('issue_summary', 'Unknown issue')

        if error_type == "none":
            logger.info("🔧 [REFINEMENT] Nothing to refine — skipping")
            return state
        
        logger.info(f"   Error type: {error_type}")
        logger.info(f"   Issue: {issue_summary}")
        
        # Different strategies for different error types
        if error_type == "hallucinated_insight":
            state = await self._refine_insights(state, critic_report)
        elif error_type in ["wrong_logic", "bad_data", "incomplete"]:
            state = await self._refine_code(state, critic_report)
        else:
            # Default: try to fix code
            state = await self._refine_code(state, critic_report)
        
        return state
    
    async def _refine_insights(self, state: ExecutionState, report: dict) -> ExecutionState:
        """Regenerate insights without re-running code"""
        logger.info("   Strategy: Regenerate insights only")
        
        prompt = f"""Previous insights were flagged as hallucinated or speculative.

# Original Question:
{state.user_query}

# Actual Data Returned:
{state.data_retrieved}

# Previous Insights (REJECTED):
{state.insights}

# Critic's Issue:
{report.get('issue_summary', 'Insights not supported by data')}

# Rules for NEW insights:
1. ONLY state facts that are DIRECTLY shown in the data above
2. If data is a single value, state that value - DO NOT speculate about trends
3. If data has comparisons, describe what you actually see
4. NO speculation, NO "likely", NO "suggests" - only what the data SHOWS
5. If the data is too limited for insights, say so honestly

# Output (strict JSON):
{{
  "insights": [
    "Fact-based insight 1",
    "Fact-based insight 2"
  ]
}}

Output:"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        try:
            import json
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                state.insights = parsed.get('insights', state.insights)
                logger.info(f"   ✅ Insights regenerated")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to parse refined insights: {e}")
        
        return state
    
    async def _refine_code(self, state: ExecutionState, report: dict) -> ExecutionState:
        """Regenerate code based on critic feedback"""
        logger.info("   Strategy: Regenerate code")
        
        # Get dataframes
        selected_ids = state.selected_file_ids or list(self.registry.files.keys())
        dfs = {fid: self.registry.get(fid).df for fid in selected_ids}
        
        # Build dataset context
        dataset_context = state.selected_files_context or ""
        
        prompt = f"""Your previous code produced flawed results. Fix it.

# Original Question:
{state.user_query}

# Available Data:
{dataset_context}

# Previous Code (FLAWED):
```python
{state.query_generated}
```

# Previous Result (PROBLEMATIC):
{state.data_retrieved}

# Critic's Diagnosis:
- Error Type: {report.get('error_type', 'unknown')}
- Issue: {report.get('issue_summary', 'Unknown')}
- Logic valid: {report.get('logic_valid', False)}
- Data valid: {report.get('data_valid', False)}
- Completeness: {report.get('completeness_valid', False)}

# Requirements for fix:
1. Address the SPECIFIC issue identified above
2. Final output MUST be in variable `result`
3. NO imports, NO unsafe operations  
4. Use exact column names from the data schema
5. Available DataFrames: {', '.join(selected_ids)}
6. If single dataset, also available as `df`

# Diagnosis checklist:
- Wrong logic → Re-examine what the question actually asks
- Bad data → Check filters aren't too strict, column names correct
- Incomplete → Make sure ALL parts of the question are answered

# Output (code only, no markdown, no explanation):"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        new_code = self._clean_code(response_text)
        
        logger.info(f"   New code generated:\n{new_code[:200]}...")
        
        # Safety check
        if not self._is_code_safe(new_code):
            logger.warning("   ⚠️ Refined code failed safety check")
            return state
        
        # Execute
        try:
            safe_globals = {"__builtins__": {"len": len, "range": range, "sum": sum, "min": min,
                     "max": max, "abs": abs, "round": round, "int": int,
                     "float": float, "str": str, "list": list, "dict": dict,
                     "set": set, "tuple": tuple, "sorted": sorted,
                     "enumerate": enumerate, "zip": zip, "map": map,
                     "filter": filter, "any": any, "all": all}, "pd": pd, **dfs}
            if len(dfs) == 1:
                safe_globals["df"] = list(dfs.values())[0].copy()
            
            exec(new_code, safe_globals)
            result = safe_globals.get("result")
            
            if result is None:
                logger.warning("   ⚠️ Refined code didn't produce `result`")
                return state
            
            # Update state with new results
            if isinstance(result, pd.DataFrame):
                state.data_retrieved = result.to_dict(orient="records")
                state.data_summary = f"{len(state.data_retrieved)} rows returned"
            elif isinstance(result, pd.Series):
                state.data_retrieved = [dict(result)]
                state.data_summary = f"Series with {len(result)} values"
            else:
                state.data_retrieved = [{"value": result}]
                state.data_summary = f"Result: {result}"
            
            state.query_generated = new_code
            state.error = ""
            logger.info(f"   ✅ Code refined successfully")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Refined code execution failed: {str(e)}")
        
        return state