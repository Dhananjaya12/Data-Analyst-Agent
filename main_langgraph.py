"""
Main entry point - Multi-agent system with LangGraph orchestration
WITH PII Redaction + Semantic Caching + Full Testing
"""

import asyncio
import os
import pandas as pd
from dotenv import load_dotenv

from data_access.csv_registry import CSVRegistry
from langgraph_orchestrator import execute_with_langgraph
from logger_config import logger
from semantic_caching import semantic_cache  # NEW
from pii_redactor import secure_pii_redactor  # NEW

# Load environment variables
load_dotenv()


async def main():
    logger.info("\n🚀 LANGGRAPH MULTI-AGENT DATA ANALYST SYSTEM")
    logger.info("WITH PII REDACTION + SEMANTIC CACHING")
    logger.info("="*70)

    token = os.getenv('GROQ_API_KEY')
    if not token:
        logger.error("❌ GROQ_API_KEY not set in .env")
        logger.info("Get free key: https://console.groq.com/keys")
        return
    
    # User ID for PII isolation
    user_id = "test_user"
    logger.info(f"👤 Running as user: {user_id}\n")
        
    # Load data
    try:
        registry = CSVRegistry()
        data_folder = "data"

        if not os.path.exists(data_folder):
            logger.info(f"❌ Folder '{data_folder}' not found")
            return

        logger.info("📂 Loading CSV files with PII redaction...")
        for file in os.listdir(data_folder):
            if file.endswith(".csv"):
                file_path = os.path.join(data_folder, file)
                try:
                    registry.register(
                        file_path=file_path,
                        description=f"Auto-loaded dataset from {file}",
                        user_id=user_id  # NEW: Pass user_id for PII
                    )
                except Exception as e:
                    logger.info(f"⚠️ Failed to load {file}: {e}")

        logger.info(f"✅ Registered {len(registry.files)} files")
        logger.info(f"Files: {list(registry.files.keys())}")
        
        # NEW: Show PII stats
        pii_count = secure_pii_redactor.get_user_mappings_count(user_id)
        logger.info(f"🔒 PII values secured: {pii_count}")
        
        logger.info("\nREGISTRY CONTEXT (what the file router sees):")
        logger.info("=" * 70)
        logger.info(registry.get_all_contexts()[:2000])

    except FileNotFoundError as e:
        logger.info(f"❌ Error loading CSV files: {e}")
        return
    
    # Clear caches for fresh test
    logger.info("\n" + "="*70)
    logger.info("🧹 CLEARING CACHES FOR FRESH TEST")
    logger.info("="*70)
    semantic_cache.clear()
    logger.info("✅ Semantic cache cleared\n")

    # Test queries covering all functionality
    test_queries = [
        # ===== SECTION 1: BASIC QUERIES (Cache Miss) =====
        {
            "section": "1️⃣ BASIC QUERIES",
            "queries": [
                "Total number of customers",
                "Average age of customers",
                "Most ordered product",
            ],
            "expected": "Cache MISS → Full execution → Results cached"
        },
        
        # # ===== SECTION 2: EXACT CACHE HITS =====
        # {
        #     "section": "2️⃣ EXACT CACHE HITS",
        #     "queries": [
        #         "Total number of customers",  # Exact repeat from Section 1
        #         "Average age of customers",   # Exact repeat from Section 1
        #     ],
        #     "expected": "Cache HIT → Instant response (both caches)"
        # },
        
        # ===== SECTION 3: SEMANTIC CACHE HITS =====
        {
            "section": "3️⃣ SEMANTIC CACHE HITS",
            "queries": [
                "How many total customers are there?",  # Similar to "Total number of customers"
                "What's the mean customer age?",        # Similar to "Average age of customers"
            ],
            "expected": "Semantic cache HIT → Instant response"
        },
        
        # # ===== SECTION 4: MULTI-TABLE JOINS =====
        # {
        #     "section": "4️⃣ MULTI-TABLE JOINS",
        #     "queries": [
        #         "Total revenue per customer",
        #         "Top 5 products by total revenue",
        #     ],
        #     "expected": "File router picks multiple files → Join logic"
        # },
        
        # # ===== SECTION 5: REFINEMENT LOOP =====
        # {
        #     "section": "5️⃣ REFINEMENT LOOP (Low Confidence)",
        #     "queries": [
        #         "City with the highest total revenue",
        #     ],
        #     "expected": "May trigger refinement if confidence < 80%"
        # },
        
        # # ===== SECTION 6: EDGE CASES =====
        # {
        #     "section": "6️⃣ EDGE CASES",
        #     "queries": [
        #         "Customers who never placed an order",
        #         "Orders where total_amount does not match calculated sum of order_items",
        #     ],
        #     "expected": "Complex logic → Handle missing data"
        # },
        
        # # ===== SECTION 7: UNANSWERABLE (Safety) =====
        # {
        #     "section": "7️⃣ UNANSWERABLE QUERIES",
        #     "queries": [
        #         "Which crypto influenced sales?",
        #         "Employee performance versus revenue",
        #     ],
        #     "expected": "Safety guard or planner rejects → No hallucination"
        # },
        
        # # ===== SECTION 8: PII RESTORATION =====
        # {
        #     "section": "8️⃣ PII RESTORATION",
        #     "queries": [
        #         "Show me customer emails with high revenue",  # If you have email column
        #     ],
        #     "expected": "LLM sees redacted → User sees original PII"
        # },
        
        # # ===== SECTION 9: FINAL CACHE CHECK =====
        # {
        #     "section": "9️⃣ FINAL CACHE VERIFICATION",
        #     "queries": [
        #         "Total revenue per customer",  # Repeat from Section 4
        #         "revenue per customer",        # Semantic variation
        #     ],
        #     "expected": "Both should hit cache (exact or semantic)"
        # },
    ]

    query_counter = 0
    total_queries = sum(len(section["queries"]) for section in test_queries)
    
    # Run all test sections
    for section_data in test_queries:
        logger.info("\n" + "="*70)
        logger.info(f"{section_data['section']}")
        logger.info("="*70)
        logger.info(f"Expected: {section_data['expected']}")
        logger.info("-"*70)
        
        for query in section_data["queries"]:
            query_counter += 1
            logger.info(f"\n📌 Query {query_counter}/{total_queries}: {query}")
            logger.info("-"*70)
            
            try:
                result = await execute_with_langgraph(
                    registry, 
                    query, 
                    user_id=user_id
                )
                
                # Display results
                logger.info(f"✅ Answer: {result['answer'][:200]}...")
                logger.info(f"📊 Confidence: {result.get('confidence', 0):.0%}")
                logger.info(f"📁 Files used: {result.get('files_used', 'N/A')}")
                logger.info(f"🔄 Refinements: {result.get('refinements', 0)}")
                
                # Cache info
                if result.get('cache_hit'):
                    if result.get('semantic_cache_hit'):
                        logger.info("🎯 STATUS: Semantic Cache HIT")
                    else:
                        logger.info("⚡ STATUS: Exact Cache HIT")
                else:
                    logger.info("❌ STATUS: Cache MISS (full execution)")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}", exc_info=True)
            
            # Rate limiting pause (except for last query)
            if query_counter < total_queries:
                logger.info("\n⏸️  Pausing 5 seconds...\n")
                await asyncio.sleep(5)
    
    # ===== FINAL STATISTICS =====
    logger.info("\n" + "="*70)
    logger.info("📊 FINAL STATISTICS")
    logger.info("="*70)
    
    # Semantic cache stats
    sem_stats = semantic_cache.get_stats()
    logger.info(f"🧠 Semantic Cache:")
    logger.info(f"   Total cached queries: {sem_stats['total_queries']}")
    logger.info(f"   Similarity threshold: {sem_stats['threshold']:.0%}")
    
    # PII stats
    pii_count = secure_pii_redactor.get_user_mappings_count(user_id)
    logger.info(f"\n🔒 PII Redaction:")
    logger.info(f"   Total secured values: {pii_count}")
    
    pii_stats = secure_pii_redactor.get_statistics()
    logger.info(f"   Total users: {pii_stats['total_users']}")
    logger.info(f"   By type: {pii_stats['mappings_by_type']}")
    
    logger.info("\n✅ All tests completed!\n")
    logger.info("="*70)
    logger.info("📋 CHECK OUTPUTS:")
    logger.info("   - outputs/llm_calls.jsonl (LLM call logs)")
    logger.info("   - outputs/query_rollup.xlsx (Metrics)")
    logger.info("   - cache/semantic_cache.json (Semantic cache)")
    logger.info("   - secure_pii.db (Encrypted PII database)")
    logger.info("="*70)


if __name__ == "__main__":
    asyncio.run(main())