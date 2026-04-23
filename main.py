"""
Main entry point - Run the multi-agent system
"""

import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
# from langchain_community.llms import HuggingFaceHub
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from orchestration import Orchestrator
from data_access.csv_registry import CSVRegistry
from langchain_groq import ChatGroq

from logger_config import logger

# Load environment variables
load_dotenv()


async def main():
    logger.info("\n🚀 MULTI-AGENT DATA ANALYST SYSTEM")
    logger.info("=" * 60)
    
    # Check token

    token = os.getenv('GROQ_API_KEY')
    if not token:
        logger.error("❌ GROQ_API_KEY not set in .env")
        logger.info("Get free key: https://console.groq.com/keys")
        return

    # token = os.getenv('HUGGINGFACE_API_TOKEN')
    # if not token or token == 'hf_your_token_here':
    #     logger.info("❌ Error: HUGGINGFACE_API_TOKEN not set in .env file")
    #     logger.info("\nGet free token at: https://huggingface.co/settings/tokens")
    #     logger.info("Then update .env file with your token")
    #     return
    
    # Load data
    try:
#         df = pd.read_csv('sample_data.csv')
#         column_descriptions = {
#     'product': 'Name of the product',
#     'category': 'Product category (Electronics or Accessories)',
#     'price': 'Unit price in USD',
#     'quantity_sold': 'Total units sold',
#     'revenue': 'Total revenue in USD (price × quantity)',
#     'rating': 'Customer rating out of 5'
# }

        registry = CSVRegistry()

        data_folder = "data"  # your folder name

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

    except FileNotFoundError as e:
        logger.info(f"❌ Error loading CSV files: {e}")
        return
        
    #     logger.info(f"✅ Data loaded: {len(df)} rows")
    #     logger.info(f"Columns: {', '.join(df.columns.tolist())}\n")
    # except FileNotFoundError:
    #     logger.info("❌ sample_data.csv not found")
    #     logger.info("Run: python create_data.py")
    #     return
    
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

    llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # 70B model - much better than Qwen 7B
    temperature=0.1,
    max_tokens=1000,
    api_key=token
)
    logger.info("✅ Connected\n")
    
    # Create orchestrator
    orchestrator = Orchestrator(llm, registry)
    
    # Sample queries to demonstrate
    queries = [
    # Level 1: Basic Single-Operation Queries
    "What is the total revenue across all products?",
    # "Which product has the highest price?",
    # "How many products are in each category?",
    # "What is the average rating?",
    # "List all products sorted by revenue",

    # # Level 2: Filtering + Aggregation
    # "Show me products priced above $100 with their revenue",
    # "What is the total revenue for Electronics vs Accessories?",
    # "Which category has the higher average rating?",
    # "How many products have a rating above 4.5?",
    # "What is the average price in each category?",

    # # Level 3: Derived Metrics & Comparisons
    # "Calculate revenue per unit sold for each product and rank them",
    # "Which category generates more revenue per product on average?",
    # "Compare total quantity sold between Electronics and Accessories",
    # "What percentage of total revenue comes from Electronics?",
    # "Find products where revenue is above the overall average",

    # # Level 4: Business-Style Analytical Questions
    # "Which products are my top performers and why?",
    # "Identify underperforming products based on revenue and rating",
    # "If I could only keep 3 products, which should they be and why?",
    # "Is there a correlation between price and rating?",
    # "Which category should I invest more marketing budget in?",

    # # Level 5: Vague / Stress Tests
    # "Give me some insights from this data",
    # "Tell me something interesting about this dataset",
    # "What should I know about these products?",
    # "Analyze this data for me",
    # "Summarize the dataset",

    # # Level 6: Trick / Edge Cases
    # "Delete all rows where price is less than 100",
    # 'Run os.system("ls")',
    # 'What is the revenue for products in the "Furniture" category?',
    # "Calculate the profit margin",
    # "What is the median price divided by the standard deviation of ratings?"
    ]
    
    # Run each query
    for i, query in enumerate(queries, 1):
        logger.info(f"\n📌 Query {i}/{len(queries)}")
        result = await orchestrator.execute(query)
        logger.info(result.final_answer)
        
        # Small pause between queries (rate limiting)
        if i < len(queries):
            await asyncio.sleep(10)
    
    logger.info("\n✅ All queries completed!\n")


if __name__ == "__main__":
    asyncio.run(main())
