"""
Main entry point - Multi-agent system with LangGraph orchestration
(Minimal changes - just swaps orchestrator for langgraph_minimal)
"""

import asyncio
import os
import pandas as pd
from dotenv import load_dotenv

from data_access.csv_registry import CSVRegistry
from langgraph_orchestrator import execute_with_langgraph
from logger_config import logger

# Load environment variables
load_dotenv()


async def main():
    logger.info("\n🚀 LANGGRAPH MULTI-AGENT DATA ANALYST SYSTEM")
    logger.info("="*60)

    token = os.getenv('GROQ_API_KEY')
    if not token:
        logger.error("❌ GROQ_API_KEY not set in .env")
        logger.info("Get free key: https://console.groq.com/keys")
        return
        
    # Load data
    try:
        registry = CSVRegistry()
        data_folder = "data"

        if not os.path.exists(data_folder):
            logger.info(f"❌ Folder '{data_folder}' not found")
            return

        for file in os.listdir(data_folder):
            if file.endswith(".csv"):
                file_path = os.path.join(data_folder, file)
                try:
                    registry.register(
                        file_path=file_path,
                        description=f"Auto-loaded dataset from {file}"
                    )
                except Exception as e:
                    logger.info(f"⚠️ Failed to load {file}: {e}")

        logger.info(f"✅ Registered {len(registry.files)} files")
        logger.info(f"Files: {list(registry.files.keys())}\n")

        logger.info("REGISTRY CONTEXT (what the file router sees):")
        logger.info("=" * 60)
        logger.info(registry.get_all_contexts()[:2000])  # First 2000 chars

    except FileNotFoundError as e:
        logger.info(f"❌ Error loading CSV files: {e}")
        return
    
    # Setup LLM
    logger.info("Connecting to Hugging Face...")

    demo_queries = [
    # Phase 1: Simple queries to warm up (2-7 seconds each)
    "Total number of customers",           # Q1: Start here - basic aggregation
    "Most ordered product",                # Q2: Single file ranking
    "Customers who never placed an order", # Q3: Show file router picks 2 files automatically
    
    # Phase 2: Multi-table joins (7-16 seconds each)
    "Total revenue per customer",          # Q4: Join + aggregation
    "Customers with more than one order and their average spend",  # Q5: Filtering + grouping
    "Top 5 products by total revenue",     # Q6: Ranking with multiple files
    
    # Phase 3: Complex queries with refinement (15-22 seconds each)
    "City with the highest total revenue", # Q7: DRAMATIC - show refinement loop (2 refinements)
    "Customers with highest number of orders",  # Q8: Show lower confidence (83%)
    
    # Phase 4: Answerability checks - show NO hallucination (2-3 seconds each)
    "Which crypto influenced sales?",      # Q9: CREDIBILITY MOMENT - instant unanswerable
    "Employee performance versus revenue", # Q10: Missing data, honest rejection
    
    # Phase 5: Data quality validation (61 seconds - hardest query)
    "Orders where total_amount does not match calculated sum of order_items",  # Q11: Complex validation
    
    # Phase 6: Cache demo (repeat Q1)
    "Total number of customers"            # Q12: Same as Q1 - shows cache hit or fast response
]

    queries = [
    # 🟢 Basic
    # "Total number of customers",
    # "Average age of customers",
    # "Count of customers with null ages",
    # "Unique cities of customers",
    # "Highest priced product",

    # 🟡 Joins + Aggregations
    "Total revenue per customer",
    "Most ordered product",
    # "Revenue per category",
    "Customers with highest number of orders",
    # "Products that were never reviewed",

    # 🔵 Multi-step reasoning
    # "Which customer spent the most overall?",
    "Top 5 products by total revenue",
    "Customers with more than one order and their average spend",
    # "Products with high sales but low ratings",
    "City with the highest total revenue",

    # 🔴 Edge cases
    # "Find orders with no matching payment record",
    "Orders where total_amount does not match calculated sum of order_items",
    "Customers who never placed an order",

    # 🧠 Ambiguity
    "Top customers",
    "Best products",

    # 🚫 Hallucination
    "Which crypto influenced sales?",
    "Employee performance versus revenue",

    # 🟣 Caching tests
    "Total revenue per customer",   # exact repeat
    "total revenue per customer",   # variation
    # "Revenue per customer",         # semantic variation
    # "Which customer spent the most overall?"  # repeat
]
    
    # Run each query
    for i, query in enumerate(queries, 1):
        logger.info(f"\n📌 Query {i}/{len(queries)}")
        
        # This now uses LangGraph internally
        result = await execute_with_langgraph(registry, query)
        
        # Small pause between queries (rate limiting)
        if i < len(queries):
            await asyncio.sleep(10)
    
    logger.info("\n✅ All queries completed!\n")


if __name__ == "__main__":
    asyncio.run(main())