"""
Metrics Tracker - CORRECTED
One Excel sheet, one row per query
"""

import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import List
import os


@dataclass
class QueryMetrics:
    """All metrics for one query"""
    query_id: str
    query_text: str
    timestamp: str
    execution_path: str = ""
    refinements: int = 0
    success: bool = True
    error: str = ""
    total_time_ms: float = 0.0
    files_used: str = ""
    llm_calls_count: int = 8  # Safety, Router, Planner, Analyst, Insights, Logic Critic, Data Critic, Insights Critic
    llm_total_input_tokens: int = 0
    llm_total_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_cache_hits: int = 0
    llm_models_used: str = ""
    code_generated: bool = False
    code_executed: bool = False
    code_execution_time_ms: float = 0.0
    code_retries: int = 0
    code_error: str = ""
    code_rows_returned: int = 0
    data_cache_hit: bool = False
    logic_score: float = 0.0
    data_score: float = 0.0
    insights_score: float = 0.0
    avg_critic_score: float = 0.0
    confidence: float = 0.0
    answer_length: int = 0
    tokens_saved: int = 0
    time_saved_ms: float = 0.0


class MetricsCollector:
    def __init__(self, output_file: str = "outputs/metrics.xlsx"):
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        self.output_file = output_file
        self.queries: List[QueryMetrics] = []
        self.query_counter = 0
    
    def start_query(self, query_text: str) -> str:
        self.query_counter += 1
        return f"Q{self.query_counter:04d}"
    
    def save_query(self, metrics: QueryMetrics):
        self.queries.append(metrics)
        self._save_excel()
        self._print_summary(metrics)
    
    def _save_excel(self):
        data = []
        for q in self.queries:
            data.append({
                'Query ID': q.query_id,
                'Query Text': q.query_text[:80],
                'Timestamp': q.timestamp,
                'Path': q.execution_path,
                'Refinements': q.refinements,
                'Success': '✅' if q.success else '❌',
                'Error': q.error,
                'Total Time (ms)': f"{q.total_time_ms:.1f}",
                'Files': q.files_used,
                'LLM Calls': q.llm_calls_count,
                'LLM Input Tokens': q.llm_total_input_tokens,
                'LLM Output Tokens': q.llm_total_output_tokens,
                'LLM Total': q.llm_total_tokens,
                'LLM Cache Hits': q.llm_cache_hits,
                'LLM Models': q.llm_models_used,
                'Code Gen': '✅' if q.code_generated else '❌',
                'Code Exec': '✅' if q.code_executed else '❌',
                'Code Time (ms)': f"{q.code_execution_time_ms:.1f}",
                'Code Retries': q.code_retries,
                'Code Rows': q.code_rows_returned,
                'Data Cache': '✅' if q.data_cache_hit else '❌',
                'Logic': f"{q.logic_score:.2f}",
                'Data': f"{q.data_score:.2f}",
                'Insights': f"{q.insights_score:.2f}",
                'Avg Score': f"{q.avg_critic_score:.2f}",
                'Confidence': f"{q.confidence:.0%}",
                'Answer Len': q.answer_length,
                'Tokens Saved': q.tokens_saved,
                'Time Saved': f"{q.time_saved_ms:.1f}",
            })
        
        if data:
            df = pd.DataFrame(data)
            df.to_excel(self.output_file, sheet_name='Queries', index=False)
            print(f"\n✅ Metrics: {self.output_file}")
    
    def _print_summary(self, q: QueryMetrics):
        print(f"\n{'='*100}")
        print(f"Q{q.query_id}: {q.query_text[:70]}")
        print(f"  Time: {q.total_time_ms:.0f}ms | Tokens: {q.llm_total_tokens} | LLM Calls: {q.llm_calls_count}")
        print(f"  Models: {q.llm_models_used}")
        print(f"  Code: {'✅' if q.code_executed else '❌'} ({q.code_rows_returned} rows) | Retries: {q.code_retries}")
        print(f"  Critics: Logic={q.logic_score:.2f}, Data={q.data_score:.2f}, Insights={q.insights_score:.2f}")
        print(f"  Confidence: {q.confidence:.0%} | Status: {'✅ Success' if q.success else '❌ Failed'}")
        print(f"{'='*100}\n")


metrics_collector = MetricsCollector()