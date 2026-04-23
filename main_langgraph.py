"""
Main entry point - Multi-agent system with LangGraph orchestration
(Minimal changes - just swaps orchestrator for langgraph_minimal)
"""

import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_groq import ChatGroq

from data_access.csv_registry import CSVRegistry
from langgraph_orchestrator import execute_with_langgraph
from logger_config import logger

# Load environment variables
load_dotenv()


async def main():
    logger.info("\n🚀 LANGGRAPH MULTI-AGENT DATA ANALYST SYSTEM")
    logger.info("="*60)
    
    # Check token
    # token = os.getenv('HUGGINGFACE_API_TOKEN')
    # if not token or token == 'hf_your_token_here':
    #     logger.info("❌ Error: HUGGINGFACE_API_TOKEN not set in .env file")
    #     logger.info("Get free token at: https://huggingface.co/settings/tokens")
    #     return

    token = os.getenv('GROQ_API_KEY_UNT')
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

    # endpoint = HuggingFaceEndpoint(
    #     repo_id="Qwen/Qwen2.5-7B-Instruct",
    #     huggingfacehub_api_token=token,
    #     temperature=0.1,
    #     max_new_tokens=300,
    #     task="conversational"
    # )
    # llm = ChatHuggingFace(llm=endpoint)

#     llm = ChatGroq(
#     model="openai/gpt-oss-120b",   # 70B model - much better than Qwen 7B
#     temperature=0.1,
#     max_tokens=1000,
#     api_key=token
# )
#     logger.info("✅ Connected\n")
    
    # Sample queries

    queries = [
    # 🟢 Basic (10)
    "Total number of customers",
    "Average age of customers",
    "Total number of customers",
    "Average age of customers",
    # "Count of orders",
    # "Average product price",
    # "List all product categories",
    # "Count of customers with null ages",
    # "Highest priced product",
    # "Lowest priced product",
    # "Number of reviews",
    # "Unique cities of customers",

    # # 🟡 Joins (10)
    # "Total revenue per customer",
    # "Total orders per city",
    # "Most ordered product",
    # "Average order value per city",
    # "Total quantity sold per product",
    # "Revenue per category",
    # "Customers with highest number of orders",
    # "Orders with failed payments",
    # "Products that were never reviewed",
    # "Customers who never placed an order",

    # # 🔵 Multi-step reasoning (10)
    # "Which customer spent the most overall?",
    # "Most popular category by quantity sold",
    # "Average rating per product category",
    # "Revenue comparison between Electronics and Fashion categories",
    # "Top 5 products by total revenue",
    # "Customers with more than one order and their average spend",
    # "Orders with missing payment but high value",
    # "Products with high sales but low ratings",
    # "City with the highest total revenue",
    # "Monthly revenue trend over time",

    # # 🔴 Hard / Orchestration (15)

    # # Data quality / edge cases
    # "Find orders with no matching payment record",
    # "Find order_items entries with invalid order_ids",
    # "Products with null prices that were still ordered",
    # "Customers with missing signup_date but who placed orders",
    # "Reviews with missing ratings",

    # # Logical reasoning
    # "Customers who only used Credit Card for payments",
    # "Customers with failed payments but high total spending",
    # "Products frequently ordered but never reviewed",
    # "Customers who ordered products but never reviewed anything",
    # "Orders where total_amount does not match calculated sum of order_items",

    # # Ambiguity / traps
    # "Top customers",
    # "Best products",
    # "Sales trends with weather impact",

    # # Noise / hallucination tests
    # "Which crypto influenced sales?",
    # "Employee performance versus revenue"
]
    # queries =     [
    # # Level 1: Basic Single-Operation Queries
    # "What is the total revenue across all products?",
    # "Which product has the highest price?",
    # # "How many products are in each category?",
    # # "What is the average rating?",
    # # "List all products sorted by revenue",

    # # # Level 2: Filtering + Aggregation
    # # "Show me products priced above $100 with their revenue",
    # # "What is the total revenue for Electronics vs Accessories?",
    # # "Which category has the higher average rating?",
    # # "How many products have a rating above 4.5?",
    # # "What is the average price in each category?",

    # # # Level 3: Derived Metrics & Comparisons
    # # "Calculate revenue per unit sold for each product and rank them",
    # # "Which category generates more revenue per product on average?",
    # # "Compare total quantity sold between Electronics and Accessories",
    # # "What percentage of total revenue comes from Electronics?",
    # # "Find products where revenue is above the overall average",

    # # # Level 4: Business-Style Analytical Questions
    # # "Which products are my top performers and why?",
    # # "Identify underperforming products based on revenue and rating",
    # # "If I could only keep 3 products, which should they be and why?",
    # # "Is there a correlation between price and rating?",
    # # "Which category should I invest more marketing budget in?",

    # # # Level 5: Vague / Stress Tests
    # # "Give me some insights from this data",
    # # "Tell me something interesting about this dataset",
    # # "What should I know about these products?",
    # # "Analyze this data for me",
    # # "Summarize the dataset",

    # # # Level 6: Trick / Edge Cases
    # # "Delete all rows where price is less than 100",
    # # 'Run os.system("ls")',
    # # 'What is the revenue for products in the "Furniture" category?',
    # # "Calculate the profit margin",
    # # "What is the median price divided by the standard deviation of ratings?"
    # ]
    
    # ===== CHANGE: Use LangGraph instead of old Orchestrator =====
    # OLD: orchestrator = Orchestrator(llm, registry)
    # NEW: Use execute_with_langgraph directly
    
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