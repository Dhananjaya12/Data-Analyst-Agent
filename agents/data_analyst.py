"""
Data Analyst Agent - Generates and executes pandas code safely
With LLM caching, data caching, and LangGraph compatibility
"""

import pandas as pd
import re
import ast
from .base import BaseAgent, ExecutionState
from logger_config import logger
from caching import cache_manager


class DataAnalystAgent(BaseAgent):
    """Executes queries and retrieves data safely"""
    
    MAX_RETRIES = 2

    def __init__(self, llm, registry, context=None):
        super().__init__("DataAnalyst", llm)
        self.registry = registry
        self.context = context

    def _is_query_safe(self, query: str) -> bool:
        """Block destructive operations"""
        forbidden = ["delete", "drop", "truncate", "overwrite"]
        return not any(word in query.lower() for word in forbidden)

    def _is_code_safe(self, code: str) -> bool:
        """AST-based safety check"""
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Block imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return False

                # Block dangerous calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["eval", "exec", "open", "__import__"]:
                            return False

                # Block dangerous modules
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id in ["os", "sys", "subprocess", "shutil"]:
                            return False

            return True
        except Exception:
            return False

    async def execute(self, state: ExecutionState) -> ExecutionState:
        logger.info(f"📊 [DATA ANALYST] Generating code...")

        # 1. SAFETY CHECK
        if not self._is_query_safe(state.user_query):
            state.error = "Destructive operations not allowed (read-only system)"
            state.data_retrieved = []
            state.data_summary = "❌ Read-only"
            return state

        # 2. CHECK DATA CACHE FIRST
        cache_key = cache_manager.get_data_cache(
            state.user_query,
            state.selected_file_ids or []
        )
        
        if cache_key:
            logger.info("✅ Using cached data result")
            state.data_retrieved = cache_key if isinstance(cache_key, list) else [cache_key]
            state.data_summary = f"[CACHED] {len(state.data_retrieved)} results"
            return state

        # 3. GET FILES
        selected_ids = getattr(state, "selected_file_ids", None)
        if not selected_ids:
            selected_ids = list(self.registry.files.keys())

        dfs = {}
        for fid in selected_ids:
            file_info = self.registry.get(fid)
            if not file_info:
                state.error = f"File '{fid}' not found"
                return state
            dfs[fid] = file_info.df

        # 4. GET DATASET CONTEXT
        dataset_context = getattr(state, "selected_files_context", None)
        if not dataset_context:
            dataset_context = self.context.build() if self.context else ""

        # 5. RETRY LOOP WITH ERROR CONTEXT
        last_error = None
        last_code = None

        for attempt in range(self.MAX_RETRIES):
            try:
                # Generate code with error context
                code = self._generate_code(state, dataset_context, last_code, last_error)
                logger.info(f"🧠 Generated Code (attempt {attempt + 1}):\n{code}\n")

                # Syntax check
                try:
                    compile(code, "<string>", "exec")
                except SyntaxError:
                    raise ValueError("Syntax error in generated code")

                # Safety check
                if not self._is_code_safe(code):
                    raise ValueError("Unsafe code detected")

                # Validate result variable exists
                if "result" not in code:
                    raise ValueError("No 'result' variable in code")

                # Execute code
                result, intermediate_results = self._execute_code(code, dfs)

                # Validate result
                self._validate_result(result)

                # Format result
                state.intermediate_results = intermediate_results
                state.execution_steps = list(intermediate_results.keys())
                state.query_generated = code
                self._format_result(state, result)

                # CACHE THE RESULT
                cache_manager.set_data_cache(
                    state.user_query,
                    state.selected_file_ids or [],
                    state.data_retrieved
                )

                logger.info(f"✅ Code executed successfully")
                return state

            except Exception as e:
                last_error = str(e)
                last_code = code if 'code' in locals() else None
                logger.info(f"❌ Attempt {attempt + 1} failed: {last_error}")

                if attempt < self.MAX_RETRIES - 1:
                    logger.info(f"🔁 Retrying with error context...\n")

        # ALL RETRIES FAILED
        state.error = f"Failed after {self.MAX_RETRIES} attempts: {last_error}"
        state.query_generated = last_code or ""
        state.data_retrieved = []
        state.data_summary = "❌ Unable to compute"
        logger.info("❌ All retries failed")
        
        return state

    def _generate_code(self, state, dataset_context, last_code=None, last_error=None):
        """Generate code with error context and LLM caching"""
        
        # Extract dataframe names
        df_names = re.findall(r"\(id:\s*(.*?)\)", dataset_context)
        df_names_str = ", ".join(df_names)
        plan = state.plan if state.plan else "No plan - analyze directly"

        # Error context for retry
        error_context = ""
        if last_error:
            error_context = f"""
# ⚠️ PREVIOUS ATTEMPT FAILED

## Code that failed:
```python
{last_code}
```

## Error:
{last_error}

## How to fix:
1. Check column names exactly (case-sensitive)
2. After groupby().sum(), use .rename() to preserve column names
3. If empty result, filter may be too strict
4. Use .astype() to convert types before math
5. After .reset_index(), column name may change → use .rename()

Rewrite correctly. Do NOT repeat the same mistake.
"""

        # Build prompt
        prompt = f"""You are a pandas code generator.

# Available DataFrames:
{dataset_context}

# DataFrame variables: {df_names_str}
{"(Single: use 'df')" if len(df_names) == 1 else f"(Multiple: use {df_names_str})"}

# Plan:
{plan}

# User Query:
{state.user_query}

# Code Requirements:
1. Store result in: `result = ...`
2. NO imports, NO file operations
3. Use EXACT column names from schema
4. Handle empty results
5. After groupby: use .rename() to preserve column names
6. For math on groups: `.rename('col').reset_index()`

{error_context}

# Code (only, no markdown):
"""

        # CHECK LLM CACHE FIRST
        cached_response = None
        if last_error is None:
            cached_response = cache_manager.get_llm_cache(
                agent="DataAnalyst",
                user_query=state.user_query,
                file_ids=state.selected_file_ids or [],
                prompt_preview=prompt[:200],
            )

        if cached_response:
            logger.info("✅ Using cached LLM response")
            code = cached_response
        else:
            logger.info("🔄 Calling LLM (new prompt)")
            response = self.llm.invoke(prompt)
            code = response.content if hasattr(response, "content") else str(response)
            if last_error is None:  # only cache first-attempt (clean) code
                cache_manager.set_llm_cache(
                    agent="DataAnalyst",
                    user_query=state.user_query,
                    file_ids=state.selected_file_ids or [],
                    response=code,
                    model=getattr(self.llm, "model", "") or getattr(self.llm, "model_name", ""),
                )

        return self._clean_code(code)

    def _clean_code(self, code):
        """Extract code from markdown if needed"""
        code = code.strip()
        
        match = re.search(r"```(?:python)?\n?(.*?)```", code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        
        return code

    def _execute_code(self, code, dfs):
        """Execute code safely with multiple dataframes"""
        
        safe_globals = {
            "__builtins__": {"len": len, "range": range, "sum": sum, "min": min,
                     "max": max, "abs": abs, "round": round, "int": int,
                     "float": float, "str": str, "list": list, "dict": dict,
                     "set": set, "tuple": tuple, "sorted": sorted,
                     "enumerate": enumerate, "zip": zip, "map": map,
                     "filter": filter, "any": any, "all": all},
            "pd": pd,
        }

        # Add all dataframes as copies
        for name, df in dfs.items():
            safe_globals[name] = df.copy()

        # If single df, expose as 'df'
        if len(dfs) == 1:
            safe_globals["df"] = list(dfs.values())[0].copy()

        # Execute code
        exec(code, safe_globals)
        result = safe_globals.get("result")

        # Capture intermediate results
        intermediate_results = {}
        for key, value in safe_globals.items():
            if key in ["pd", "__builtins__", "result"] or key in dfs:
                continue

            if isinstance(value, pd.DataFrame):
                if len(value) <= 1000:
                    intermediate_results[key] = value.to_dict(orient="records")
            elif isinstance(value, pd.Series):
                intermediate_results[key] = dict(value)
            elif isinstance(value, (int, float, str)):
                intermediate_results[key] = value

        return result, intermediate_results

    def _validate_result(self, result):
        """Validate result is usable"""
        if result is None:
            raise ValueError("Code didn't create 'result' variable")

        if isinstance(result, pd.DataFrame) and len(result) == 0:
            raise ValueError("Result is empty DataFrame - filter too strict?")

    def _format_result(self, state, result):
        """Format result into state"""
        if isinstance(result, pd.DataFrame):
            state.data_retrieved = result.to_dict(orient="records")
            state.data_summary = f"{len(state.data_retrieved)} rows"
        elif isinstance(result, pd.Series):
            state.data_retrieved = [dict(result)]
            state.data_summary = f"Series: {len(result)} values"
        else:
            state.data_retrieved = [{"value": result}]
            state.data_summary = f"Result: {result}"