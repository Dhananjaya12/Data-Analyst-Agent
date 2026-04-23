"""
Orchestration - Coordinates multi-agent workflow (Multi-file + Safety + Logging)
"""

from data_access.csv_registry import CSVRegistry
from agents.file_router import FileRouterAgent
from agents.planner import PlannerAgent
from agents.data_analyst import DataAnalystAgent
from agents.insight_generator import InsightGeneratorAgent
from agents.critic import CriticAgent
from agents.context_builder import DatasetContext
from agents.base import ExecutionState, StopExecution
from agents.safety_guard import SafetyGuardAgent
from logger_config import logger


class Orchestrator:
    """Coordinates the multi-agent workflow"""

    def __init__(self, llm, registry: CSVRegistry):
        self.llm = llm
        self.registry = registry

        # ✅ Safety layer (kept)
        self.safety_guard = SafetyGuardAgent(llm)

        # ✅ Context (optional, for fallback / future use)
        self.context = DatasetContext(registry)

        # ✅ Multi-file routing
        self.file_router = FileRouterAgent(llm, registry)

        # ✅ Agents
        self.planner = PlannerAgent(llm)  # now file-aware via state
        self.data_analyst = DataAnalystAgent(llm, registry)
        self.insight_generator = InsightGeneratorAgent(llm)
        self.critic = CriticAgent(llm)

    async def execute(self, query: str) -> ExecutionState:
        """Execute the full pipeline"""

        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {query}")
        logger.info('='*60)

        state = ExecutionState(user_query=query)

        try:
            # 🛡️ 0. Safety Guard (kept)
            state = await self.safety_guard.execute(state)

            # 📂 1. File Routing (NEW)
            state = await self.file_router.execute(state)

            if getattr(state, "error", None):
                raise StopExecution(state)

            # 🧠 2. Planning
            state = await self.planner.execute(state)

            # 📊 3. Data Analysis (multi-file aware)
            state = await self.data_analyst.execute(state)

            # 💡 4. Insights
            state = await self.insight_generator.execute(state)

            # ✅ 5. Critic / Validation
            state = await self.critic.execute(state)

            # 🧾 6. Final formatting (ENHANCED)
            insights_text = '\n'.join(state.insights[:3]) if state.insights else 'N/A'
            validation = '✅ Valid' if state.is_valid else '⚠️ Review needed'
            files_used = ', '.join(getattr(state, "selected_file_ids", []))

            state.final_answer = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 FINAL ANSWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Question: {state.user_query}

Files Used: {files_used}

Answer: Derived from processed dataset ({len(state.data_retrieved)} records)

Code Used:
{state.query_generated}

Insights:
{insights_text}

Validation: {validation}
Confidence: {state.confidence * 100:.0f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        except StopExecution as e:
            logger.info("⛔ Execution stopped safely")

            state = e.state
            state.final_answer = f"""
❌ Execution Stopped

Reason:
{state.error}
"""
            return state

        except Exception as e:
            logger.error(f"💥 Unexpected error: {str(e)}")

            state.error = str(e)
            state.final_answer = f"❌ Unexpected Error: {str(e)}"
            return state

        return state