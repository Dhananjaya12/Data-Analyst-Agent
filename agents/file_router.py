"""
File Router Agent - Picks the right CSV file(s) for a query
"""

import json
import re
from .base import BaseAgent, ExecutionState


class FileRouterAgent(BaseAgent):
    """Decides which registered CSV file(s) are relevant to the query"""
    
    def __init__(self, llm, registry):
        super().__init__("FileRouter", llm)
        self.registry = registry
    
    async def execute(self, state: ExecutionState) -> ExecutionState:
        print(f"🗂️  [FILE ROUTER] Selecting relevant files...")
        
        files = self.registry.list_files()
        
        # If no files
        if len(files) == 0:
            state.error = "No files registered"
            return state
        
        # If only one file, skip routing
        if len(files) == 1:
            state.selected_file_ids = [files[0].file_id]
            state.selected_files_context = files[0].build_context()
            print(f"✅ Only one file: {files[0].file_id}")
            return state
        
        # Multiple files - use LLM to route
        all_contexts = self.registry.get_all_contexts()
        
        prompt = f"""You are a file routing expert. Pick the MINIMUM files needed.

# CRITICAL RULE: 
Default to ONE file. Only pick multiple if the query REQUIRES columns from different files (i.e., a join is mandatory).

# Available files (with columns and sample data):
{all_contexts}

# Decision Algorithm (follow exactly):

STEP 1: Identify required columns from the query
   - What column(s) does the answer need?
   - Example: "total revenue" needs a `revenue` column
   - Example: "revenue by category" needs `revenue` AND `category` columns

STEP 2: Find the SMALLEST set of files containing those columns
   - If ALL required columns are in ONE file → pick that file
   - If columns are spread across files → pick only the necessary files

STEP 3: Verify with sample data
   - Check the SAMPLE DATA shown for each file
   - Confirm the columns actually contain what you expect

# Examples:

Query: "What is the total revenue?"
Analysis: Need `revenue` column. Check files → found in sales_data only.
→ {{"selected_files": ["sales_data"], "reasoning": "revenue column exists in sales_data alone"}}

Query: "Revenue per product category"  
Analysis: Need `revenue` (in sales_data) AND `category` (in products_data). Must join.
→ {{"selected_files": ["sales_data", "products_data"], "reasoning": "revenue in sales, category in products - join required"}}

Query: "How many customers do we have?"
Analysis: Need customer count. Check files → customer records in customers_data.
→ {{"selected_files": ["customers_data"], "reasoning": "customer count from customers file only"}}

# Output (strict JSON):
{{
  "required_columns": ["col1", "col2"],
  "selected_files": ["file_id"],
  "reasoning": "why these files and not others"
}}

# User Query:
{state.user_query}

# Output:"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        valid_ids = []
        reasoning = ""
        
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                candidate_ids = parsed.get("selected_files", [])
                required_cols = parsed.get("required_columns", [])
                reasoning = parsed.get("reasoning", "")
                
                # Validate IDs against registry
                valid_ids = [fid for fid in candidate_ids if fid in self.registry.files]
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"⚠️ Failed to parse LLM output: {e}")
        
        # Fallback: if parsing failed or no valid IDs, use all files
        if not valid_ids:
            print("⚠️ Router output unclear, using all files as fallback")
            valid_ids = [f.file_id for f in files]
        
        # Update state
        state.selected_file_ids = valid_ids
        state.selected_files_context = "\n\n".join(
            self.registry.get(fid).build_context() for fid in valid_ids
        )
        
        if reasoning:
            print(f"✅ Selected files: {valid_ids} | Reasoning: {reasoning}")
        else:
            print(f"✅ Selected files: {valid_ids}")
        
        return state