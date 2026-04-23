"""
LangGraph Orchestrator with Ensemble Critics, Self-Healing Loop,
rich progress callbacks, and clean structured response.
"""

import uuid
from typing import Literal
from langgraph.graph import StateGraph, END

from agents.base import ExecutionState
from agents.safety_guard import SafetyGuardAgent
from agents.file_router import FileRouterAgent
from agents.planner import PlannerAgent
from agents.data_analyst import DataAnalystAgent
from agents.insight_generator import InsightGeneratorAgent
from agents.critic_ensemble import LogicCritic, DataCritic, InsightsCritic
from agents.refinement import RefinementAgent
from logger_config import logger

from observability import tracker, observe_agent


MAX_REFINEMENTS = 2


def route_after_ensemble(state) -> Literal["finalize", "refine"]:
    if isinstance(state, dict):
        confidence = state.get('confidence', 0)
        refine_count = state.get('refinement_count', 0)
    else:
        confidence = getattr(state, 'confidence', 0)
        refine_count = getattr(state, 'refinement_count', 0)

    logger.info(f"🔀 ROUTE: conf={confidence:.0%}, attempts={refine_count}/{MAX_REFINEMENTS}")

    if confidence >= 0.8:
        logger.info("   → Finalize (quality met)")
        return "finalize"
    elif refine_count < MAX_REFINEMENTS:
        logger.info(f"   → Refine (attempt {refine_count + 1}/{MAX_REFINEMENTS})")
        return "refine"
    else:
        logger.info("   → Finalize (max refinements, ship best effort)")
        return "finalize"


def make_nodes(registry, status_callback=None):
    from llm_config import (
        get_fast_llm, get_reasoning_llm, get_insights_llm,
        get_code_llm, get_critic_llm,
    )

    fast_llm = get_fast_llm()
    reasoning_llm = get_reasoning_llm()
    insights_llm = get_insights_llm()
    code_llm = get_code_llm()
    critic_llm = get_critic_llm()

    safety          = SafetyGuardAgent(fast_llm)
    file_router     = FileRouterAgent(fast_llm, registry)
    planner         = PlannerAgent(reasoning_llm)
    analyst         = DataAnalystAgent(code_llm, registry)
    insights        = InsightGeneratorAgent(insights_llm)
    refinement      = RefinementAgent(code_llm, registry)
    logic_critic    = LogicCritic(critic_llm)
    data_critic     = DataCritic(critic_llm)
    insights_critic = InsightsCritic(critic_llm)

    # --- helpers --------------------------------------------------------

    def _emit(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    def _get(state, key, default=""):
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    def _truncate(s, n=400):
        s = str(s)
        return s if len(s) <= n else s[:n] + "…"

    # --- nodes ----------------------------------------------------------

    async def safety_node(state):
        _emit("🛡️  **Safety Guard** checking query...")
        with observe_agent("SafetyGuard"):
            result = await safety.execute(state)
        _emit("   ✅ `SAFE` — query approved")
        return result

    async def file_router_node(state):
        _emit("🗂️  **File Router** selecting files...")
        with observe_agent("FileRouter"):
            result = await file_router.execute(state)
        files = _get(result, "selected_file_ids", [])
        _emit(f"   ✅ Selected: `{', '.join(files) if files else 'none'}`")
        return result

    async def planner_node(state):
        _emit("🧠 **Planner** breaking down the query...")
        with observe_agent("Planner"):
            result = await planner.execute(state)
        plan = _get(result, "plan", "")
        _emit(f"   📋 Plan:\n```\n{_truncate(plan)}\n```")
        return result

    async def analyst_node(state):
        _emit("📊 **Data Analyst** generating code...")
        with observe_agent("DataAnalyst"):
            result = await analyst.execute(state)
        code    = _get(result, "query_generated", "")
        summary = _get(result, "data_summary", "")
        data    = _get(result, "data_retrieved", [])

        if code:
            _emit(f"   💻 Code:\n```python\n{_truncate(code)}\n```")
        _emit(f"   📈 Result: `{summary}`")
        if data:
            _emit(f"   👀 Preview: `{_truncate(str(data[:3]), 300)}`")
        return result

    async def insights_node(state):
        _emit("💡 **Insight Generator** analyzing results...")
        with observe_agent("InsightGenerator"):
            result = await insights.execute(state)
        ins = _get(result, "insights", [])
        if ins:
            bullets = "\n".join(f"      • {i}" for i in ins[:5])
            _emit(f"   💡 Insights:\n{bullets}")
        else:
            _emit("   ⚠️ No insights generated")
        return result

    async def logic_critic_node(state):
        # _emit("🔍 **Logic Critic** reviewing code...")
        with observe_agent("LogicCritic"):
            result = await logic_critic.execute(state)
        score = result.get("logic_score", 0.0) if isinstance(result, dict) else 0.0
        issue = result.get("logic_issue", "")    if isinstance(result, dict) else ""
        # _emit(f"   ⚖️ Logic: `{score:.2f}` — {issue}")
        return result

    async def data_critic_node(state):
        # _emit("🔍 **Data Critic** reviewing data...")
        with observe_agent("DataCritic"):
            result = await data_critic.execute(state)
        score = result.get("data_score", 0.0) if isinstance(result, dict) else 0.0
        issue = result.get("data_issue", "")   if isinstance(result, dict) else ""
        # _emit(f"   ⚖️ Data: `{score:.2f}` — {issue}")
        return result

    async def insights_critic_node(state):
        # _emit("🔍 **Insights Critic** reviewing insights...")
        with observe_agent("InsightsCritic"):
            result = await insights_critic.execute(state)
        score = result.get("insights_score", 0.0) if isinstance(result, dict) else 0.0
        issue = result.get("insights_issue", "")  if isinstance(result, dict) else ""
        # _emit(f"   ⚖️ Insights: `{score:.2f}` — {issue}")
        return result

    async def merge_critics_node(state):
        logic_score    = _get(state, 'logic_score', 0.5)
        data_score     = _get(state, 'data_score', 0.5)
        insights_score = _get(state, 'insights_score', 0.5)
        logic_issue    = _get(state, 'logic_issue', '')
        data_issue     = _get(state, 'data_issue', '')
        insights_issue = _get(state, 'insights_issue', '')

        # Announce that critics ran (batched)
        _emit("🔍 **Critics** reviewing in parallel...")
        _emit(f"   ⚖️ Logic:    `{logic_score:.2f}` — {logic_issue}")
        _emit(f"   ⚖️ Data:     `{data_score:.2f}` — {data_issue}")
        _emit(f"   ⚖️ Insights: `{insights_score:.2f}` — {insights_issue}")

        scores = [logic_score, data_score, insights_score]
        confidence = sum(scores) / len(scores)
        pass_count = sum(1 for s in scores if s >= 0.7)

        score_map = {
            "wrong_logic":          logic_score,
            "bad_data":             data_score,
            "hallucinated_insight": insights_score,
        }
        error_type = min(score_map, key=score_map.get)
        if score_map[error_type] >= 0.7:
            error_type = "none"

        issue_summary = {
            "wrong_logic":          logic_issue,
            "bad_data":             data_issue,
            "hallucinated_insight": insights_issue,
            "none":                 "",
        }.get(error_type, "")

        _emit(f"⚖️ **Consensus**: {pass_count}/3 passed, confidence={confidence:.0%}")
        if error_type != "none":
            _emit(f"   ⚠️ Primary issue: `{error_type}` — {issue_summary}")

        return {
            "confidence": confidence,
            "is_valid": pass_count >= 2,
            "critic_report": {
                "logic_score":         logic_score,
                "data_score":          data_score,
                "insights_score":      insights_score,
                "overall_confidence":  confidence,
                "pass_count":          pass_count,
                "error_type":          error_type,
                "issue_summary":       issue_summary,
                "logic_valid":         logic_score    >= 0.7,
                "data_valid":          data_score     >= 0.7,
                "completeness_valid":  insights_score >= 0.7,
            },
        }

    async def refinement_node(state):
        if isinstance(state, dict):
            state['refinement_count'] = state.get('refinement_count', 0) + 1
            count = state['refinement_count']
        else:
            state.refinement_count = getattr(state, 'refinement_count', 0) + 1
            count = state.refinement_count

        _emit(f"🔧 **Refinement** attempt {count}/{MAX_REFINEMENTS}...")
        with observe_agent("Refinement"):
            result = await refinement.execute(state)

        new_code = _get(result, "query_generated", "")
        if new_code:
            _emit(f"   💻 Refined code:\n```python\n{_truncate(new_code, 300)}\n```")
        return result

    async def finalize_node(state):
        _emit("✅ **Finalizing** response...")
        return state

    return {
        "safety": safety_node,
        "file_router": file_router_node,
        "planner": planner_node,
        "analyst": analyst_node,
        "insights": insights_node,
        "logic_critic": logic_critic_node,
        "data_critic": data_critic_node,
        "insights_critic": insights_critic_node,
        "merge_critics": merge_critics_node,
        "refinement": refinement_node,
        "finalize": finalize_node,
    }


def create_analysis_graph(registry, status_callback=None):
    nodes = make_nodes(registry, status_callback=status_callback)
    workflow = StateGraph(ExecutionState)

    workflow.add_node("safety_guard", nodes["safety"])
    workflow.add_node("file_router", nodes["file_router"])
    workflow.add_node("planner", nodes["planner"])
    workflow.add_node("data_analyst", nodes["analyst"])
    workflow.add_node("insight_generator", nodes["insights"])
    workflow.add_node("logic_critic", nodes["logic_critic"])
    workflow.add_node("data_critic", nodes["data_critic"])
    workflow.add_node("insights_critic", nodes["insights_critic"])
    workflow.add_node("merge_critics", nodes["merge_critics"])
    workflow.add_node("refinement", nodes["refinement"])
    workflow.add_node("finalize", nodes["finalize"])

    workflow.add_edge("safety_guard", "file_router")
    workflow.add_edge("file_router", "planner")
    workflow.add_edge("planner", "data_analyst")
    workflow.add_edge("data_analyst", "insight_generator")

    # Fan out: parallel critics
    workflow.add_edge("insight_generator", "logic_critic")
    workflow.add_edge("insight_generator", "data_critic")
    workflow.add_edge("insight_generator", "insights_critic")

    # Fan in
    workflow.add_edge("logic_critic", "merge_critics")
    workflow.add_edge("data_critic", "merge_critics")
    workflow.add_edge("insights_critic", "merge_critics")

    workflow.add_conditional_edges(
        "merge_critics",
        route_after_ensemble,
        {"finalize": "finalize", "refine": "refinement"},
    )

    # Refinement also fans out to the three critics in parallel
    workflow.add_edge("refinement", "logic_critic")
    workflow.add_edge("refinement", "data_critic")
    workflow.add_edge("refinement", "insights_critic")

    workflow.add_edge("finalize", END)
    workflow.set_entry_point("safety_guard")

    return workflow.compile()


async def execute_with_langgraph(registry, query: str, context: str = "",
                                  status_callback=None) -> dict:
    """
    Returns a dict:
        {
          "answer":       str,       # clean, user-facing answer (insights only)
          "insights":     List[str],
          "confidence":   float,
          "files_used":   str,
          "is_valid":     bool,
          "refinements":  int,
          "cache_hit":    bool,
        }
    """
    from caching import cache_manager

    def _status(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    query_id = f"Q-{uuid.uuid4().hex[:8]}"
    tracker.start_query(query_id, query)

    # --- Full-response cache short-circuit -----------------------------
    file_ids = sorted(registry.files.keys())
    cached = cache_manager.get_full_response(query, file_ids)
    if cached:
        _status("⚡ **Cache hit** — returning cached answer instantly")
        tracker.end_query(
            status="ok", confidence=cached.get("confidence", 1.0),
            refinements=0, files_used=", ".join(file_ids),
            extra={"full_response_cache_hit": True},
        )
        return {**cached, "cache_hit": True}

    try:
        state = ExecutionState(user_query=query)
        state.conversation_context = context

        app = create_analysis_graph(registry, status_callback=status_callback)
        result = await app.ainvoke(state)

        is_dict = isinstance(result, dict)
        def g(k, d=None):
            return result.get(k, d) if is_dict else getattr(result, k, d)

        insights    = g("insights", []) or []
        is_valid    = bool(g("is_valid", False))
        confidence  = g("confidence", 0) or 0
        refinements = g("refinement_count", 0) or 0
        files_used  = ", ".join(g("selected_file_ids", []) or [])

        # Clean user-facing answer: JUST the insights, no metadata spam
        if insights:
            answer = "\n\n".join(insights[:5])
        else:
            answer = "I couldn't generate insights for this query. Please try rephrasing."

        response = {
            "answer":      answer,
            "insights":    insights,
            "confidence":  confidence,
            "files_used":  files_used,
            "is_valid":    is_valid,
            "refinements": refinements,
            "cache_hit":   False,
        }

        # Cache only if we got a confidently-good answer
        if confidence >= 0.7 and insights:
            cache_manager.set_full_response(query, file_ids, response)

        tracker.end_query(
            status="ok", confidence=confidence,
            refinements=refinements, files_used=files_used,
        )
        return response

    except Exception as e:
        tracker.end_query(status="error", error=str(e))
        raise