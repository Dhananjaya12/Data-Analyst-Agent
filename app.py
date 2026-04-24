"""
CSV Chat Assistant — Streamlit UI with analysis next to each message
Powered by GROQ + LangGraph multi-agent orchestration.
"""

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from data_access.csv_registry import CSVRegistry
from langgraph_orchestrator import execute_with_langgraph
from caching import cache_manager
from logger_config import logger
from observability import wrap_llm


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Data Chat",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM STYLES
# ============================================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    section[data-testid="stSidebar"] .stHeadingContainer {
        padding-top: 0;
    }
    .file-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .file-card-name {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .file-card-meta {
        color: rgba(128, 128, 128, 0.9);
        font-size: 0.75rem;
        margin-top: 2px;
    }
    .sample-chip {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 16px;
        background: rgba(100, 150, 255, 0.08);
        border: 1px solid rgba(100, 150, 255, 0.2);
        cursor: pointer;
        font-size: 0.85rem;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    [data-testid="stChatMessage"] {
        padding: 0.75rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# STATE
# ============================================================================
os.makedirs("data", exist_ok=True)

if "registry" not in st.session_state:
    st.session_state.registry = CSVRegistry()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = set()
if "cache" not in st.session_state:
    st.session_state.cache = cache_manager
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

registry = st.session_state.registry
cache = st.session_state.cache


# ============================================================================
# LLM (fallback chat, used when no files are uploaded)
# ============================================================================
@st.cache_resource
def get_chat_llm():
    load_dotenv()
    token = os.getenv("GROQ_API_KEY")
    if not token:
        st.error("GROQ_API_KEY not set in .env")
        st.stop()
    return wrap_llm(ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1000,
        api_key=token,
    ))


try:
    chat_llm = get_chat_llm()
except Exception as e:
    st.error(f"LLM init failed: {e}")
    st.stop()


# ============================================================================
# HELPER: Get file data with timestamp to force refresh
# ============================================================================
def get_file_data(filepath):
    """Read file and return data + timestamp for cache busting"""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Data Chat")
    st.caption("Ask questions about your CSVs in natural language")
    st.divider()

    # --- File section ---
    st.markdown("##### Your data")

    if registry.files:
        for fid, file_info in registry.files.items():
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f"""
                <div class="file-card">
                    <div class="file-card-name">{fid}</div>
                    <div class="file-card-meta">{file_info.row_count:,} rows · {len(file_info.df.columns)} cols</div>
                </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.write("")
                if st.button("✕", key=f"del_{fid}", help=f"Remove {fid}"):
                    registry.remove(fid)
                    st.session_state.uploaded_file_names.discard(fid)
                    st.rerun()
    else:
        st.caption("_No files yet_")

    # --- Upload ---
    with st.popover("➕ Add CSV", use_container_width=True):
        uploaded_files = st.file_uploader(
            "Select CSV files",
            accept_multiple_files=True,
            type=["csv"],
            label_visibility="collapsed",
        )

        if uploaded_files:
            for f in uploaded_files:
                file_id = os.path.splitext(f.name)[0]
                if file_id in st.session_state.uploaded_file_names:
                    continue
                file_path = f"data/{f.name}"
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as out:
                        out.write(f.read())
                try:
                    if file_id not in registry.files:
                        registry.register(file_path, description=f"Data from {f.name}", file_id=file_id)
                    st.session_state.uploaded_file_names.add(file_id)
                except Exception as e:
                    logger.error(f"Upload error: {e}")
                    st.error(f"Could not load {f.name}")

    st.divider()

    # --- Conversation controls ---
    if st.session_state.messages:
        if st.button("🧹 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            cache.conversation = []
            st.rerun()

    # --- Developer info with downloads ---
    with st.expander("Developer"):
        stats = cache.get_stats()
        st.caption(f"Cached queries: {stats['data_cache_size']}")
        st.caption(f"LLM prompts cached: {stats['llm_cache_size']}")
        
        st.divider()
        st.markdown("**📊 Export Metrics**")
        
        col1, col2 = st.columns(2)
        
        # Download LLM Calls JSONL
        with col1:
            jsonl_data = get_file_data('outputs/llm_calls.jsonl')
            if jsonl_data:
                st.download_button(
                    label="📋 LLM Calls",
                    data=jsonl_data,
                    file_name="llm_calls.jsonl",
                    mime="application/json",
                    use_container_width=True,
                    key=f"download_jsonl_{len(st.session_state.messages)}"  # Force refresh
                )
            else:
                st.caption("_No calls logged_")
        
        # Download Query Rollup Excel
        with col2:
            xlsx_data = get_file_data('outputs/query_rollup.xlsx')
            if xlsx_data:
                st.download_button(
                    label="📈 Query Rollup",
                    data=xlsx_data,
                    file_name="query_rollup.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=f"download_xlsx_{len(st.session_state.messages)}"  # Force refresh
                )
            else:
                st.caption("_No metrics yet_")
        
        st.divider()
        if st.button("🧹 Reset caches", use_container_width=True):
            cache.clear_all()
            st.rerun()


# ============================================================================
# MAIN AREA - Render chat with analysis expanders NEXT TO each message
# ============================================================================
def render_empty_state():
    """Shown when no chat has started yet."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align: center; font-weight: 600;'>What would you like to know?</h2>",
        unsafe_allow_html=True,
    )

    if not registry.files:
        st.markdown(
            "<p style='text-align: center; color: rgba(128,128,128,0.9);'>"
            "Upload a CSV from the sidebar to start analyzing, or just chat below."
            "</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='text-align: center; color: rgba(128,128,128,0.9);'>"
            f"Analyzing {len(registry.files)} file{'s' if len(registry.files) != 1 else ''}. "
            "Try one of these:"
            "</p>",
            unsafe_allow_html=True,
        )

        samples = [
            "What's in this data?",
            "Show me summary statistics",
            "Find any anomalies",
            "What are the top trends?",
        ]
        cols = st.columns(len(samples))
        for i, sample in enumerate(samples):
            with cols[i]:
                if st.button(sample, key=f"sample_{i}", use_container_width=True):
                    st.session_state.pending_query = sample
                    st.rerun()


# Render chat history or empty state
if not st.session_state.messages:
    render_empty_state()
else:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
        
        # NEW: If this is an assistant message AND it has analysis, show it right after
        if msg["role"] == "assistant" and msg.get("progress"):
            query_label = msg.get("query", "Query")[:50]
            with st.expander(f"📊 Analysis: {query_label}", expanded=False):
                st.markdown(msg["progress"])


# ============================================================================
# INPUT + HANDLER
# ============================================================================
prompt = st.chat_input("Ask about your data...") or st.session_state.pending_query
st.session_state.pending_query = None

if prompt:
    # Persist and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    cache.add_to_conversation("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant turn
    with st.chat_message("assistant"):
        status = st.status("Thinking...", state="running", expanded=False)
        answer_slot = st.empty()
        meta_slot = st.empty()

        answer = ""
        meta_line = ""
        result = None
        progress_log = []  # Collect progress for analysis

        try:
            if not registry.files:
                # --- Fallback chat mode ---
                history = cache.get_conversation_text(last_n=6)
                chat_prompt = (
                    "You are a helpful assistant. The user has not uploaded any CSV "
                    "files. Answer conversationally. If they ask data questions, "
                    "suggest they upload a CSV.\n\n"
                    f"{history}\nuser: {prompt}\nassistant:"
                )
                response = chat_llm.invoke(chat_prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                status.update(label="Done", state="complete")

            else:
                # --- LangGraph analysis mode ---
                status.update(label="Analyzing...", state="running")
                context = cache.get_conversation_text(last_n=4)

                def on_status(msg: str):
                    progress_log.append(msg)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        execute_with_langgraph(
                            registry, prompt, context=context,
                            status_callback=on_status,
                        )
                    )
                finally:
                    loop.close()

                # Render progress into the (collapsed) status expander
                with status:
                    for msg in progress_log:
                        st.markdown(msg)

                answer = result["answer"]

                # Compact status label
                if result.get("cache_hit"):
                    status.update(label="⚡ Instant (cached)", state="complete")
                else:
                    conf = result.get("confidence", 0)
                    status.update(label=f"Done · {conf:.0%} confidence", state="complete")

                # Metadata line
                meta_parts = []
                if result.get("files_used"):
                    meta_parts.append(f"📁 {result['files_used']}")
                if result.get("cache_hit"):
                    meta_parts.append("⚡ cached")
                elif result.get("refinements", 0) > 0:
                    meta_parts.append(f"🔧 refined {result['refinements']}×")
                meta_line = "  ·  ".join(meta_parts)

            answer_slot.markdown(answer)
            if meta_line:
                meta_slot.caption(meta_line)

        except Exception as e:
            answer = f"Something went wrong: `{e}`"
            answer_slot.markdown(answer)
            status.update(label="Error", state="error")
            logger.error(f"Query error: {e}", exc_info=True)

    # Persist - NOW INCLUDE progress in the message
    progress_text = "\n".join(progress_log) if progress_log else ""
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": meta_line,
        "query": prompt,           # Store original query
        "progress": progress_text  # Store analysis/progress
    })
    cache.add_to_conversation("assistant", answer)