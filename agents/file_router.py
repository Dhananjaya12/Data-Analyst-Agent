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
            state.answerability = "ANSWERABLE"
            state.answerability_reason = ""
            print(f"✅ Only one file: {files[0].file_id}")
            return state
        
        # Multiple files - use LLM to route
        all_contexts = self.registry.get_all_contexts()
        
        prompt = f"""You are a file routing expert. For the query:
1. Pick the MINIMUM files needed to answer it
2. Determine if the query is answerable from available data

# CRITICAL RULE: 
Default to ONE file. Only pick multiple if the query REQUIRES columns from different files (i.e., a join is mandatory).

# Available files (with columns and sample data):
{all_contexts}

# TASK 1: File Selection (follow exactly):

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

# TASK 2: Answerability Check

After selecting files, ask: "Can this query be answered using ONLY the selected files?"

# Output (strict JSON):
{{
  "selected_files": ["file_id"],
  "reasoning": "why these files",
  "answerability": "ANSWERABLE" | "UNANSWERABLE",
  "answerability_reason": "brief explanation"
}}

# User Query:
{state.user_query}

# Output:"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        valid_ids = []
        reasoning = ""
        answerability = "ANSWERABLE"
        answerability_reason = ""
        
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                candidate_ids = parsed.get("selected_files", [])
                reasoning = parsed.get("reasoning", "")
                answerability = parsed.get("answerability", "ANSWERABLE")
                answerability_reason = parsed.get("answerability_reason", "")
                
                # Validate IDs against registry
                valid_ids = [fid for fid in candidate_ids if fid in self.registry.files]
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"⚠️ Failed to parse LLM output: {e}")
        
        # Fallback: if parsing failed or no valid IDs, use all files (only if answerable)
        if not valid_ids and answerability == "ANSWERABLE":
            print("⚠️ Router output unclear, using all files as fallback")
            valid_ids = [f.file_id for f in files]
        
        # Update state with file selection AND answerability
        state.selected_file_ids = valid_ids
        state.answerability = answerability
        state.answerability_reason = answerability_reason
        
        if valid_ids:
            state.selected_files_context = "\n\n".join(
                self.registry.get(fid).build_context() for fid in valid_ids
            )
        
        # Log the routing decision
        if answerability == "UNANSWERABLE":
            print(f"❌ Query is UNANSWERABLE: {answerability_reason}")
        else:
            if reasoning:
                print(f"✅ Selected files: {valid_ids} | Reasoning: {reasoning}")
            else:
                print(f"✅ Selected files: {valid_ids}")
        
        return state