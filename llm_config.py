"""
Centralized LLM configuration - task-optimized parameters
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from observability import wrap_llm

load_dotenv()

GROQ_KEY = os.getenv('GROQ_API_KEY')


def get_fast_llm():
    """Small & fast - for classification (safety, file routing)"""
    return wrap_llm(ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,       # Deterministic - classification needs consistency
        max_tokens=300,
        timeout=30, 
        api_key=GROQ_KEY
    ))

def get_reasoning_llm():
    """Planning - needs structured thinking"""
    return wrap_llm(ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,       # Low - plans should be deterministic
        max_tokens=600,
        timeout=30, 
        api_key=GROQ_KEY
    ))


def get_insights_llm():
    """Insights - needs some variety but grounded in facts"""
    return wrap_llm(ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,       # Slight variety, but fact-focused
        max_tokens=500,
        timeout=30, 
        api_key=GROQ_KEY
    ))


def get_code_llm():
    """Code generation - must be precise"""
    return wrap_llm(ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,       # Zero creativity - code must be exact
        max_tokens=1500,
        timeout=30, 
        api_key=GROQ_KEY
    ))


def get_critic_llm():
    """Critic - needs strict evaluation, not lazy pattern matching"""
    return wrap_llm(ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=300,        # critics only emit a short JSON
        timeout=30, 
        api_key=GROQ_KEY,
    ))