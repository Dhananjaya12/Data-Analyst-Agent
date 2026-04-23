# caching.py
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, List

from observability import record_cache_hit


class CacheManager:
    def __init__(self):
        self.llm_cache = {}
        self.data_cache = {}
        self.conversation = []
        self.cache_ttl = 3600


    # ---------- FULL-RESPONSE CACHE ----------
    def _response_key(self, user_query: str, file_ids: list) -> str:
        """Semantic key: same query + same files = same answer."""
        files_part = ",".join(sorted(file_ids or []))
        key_str = f"response||{user_query.strip().lower()}||{files_part}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_full_response(self, user_query: str, file_ids: list) -> Optional[str]:
        """Return a cached full answer for (query, files), or None."""
        import time
        t0 = time.perf_counter()
        key = self._response_key(user_query, file_ids)
        if key in self.data_cache:   # reuse data_cache dict for simplicity
            entry = self.data_cache[key]
            if datetime.now() < entry['expires'] and entry.get('kind') == 'full_response':
                latency_ms = (time.perf_counter() - t0) * 1000
                record_cache_hit(
                    model="full-response-cache",
                    prompt_preview=user_query,
                    cached_response=str(entry['response'])[:200],
                    cache_type="full_response",
                    latency_ms=latency_ms,
                )
                return entry['response']
        return None

    def set_full_response(self, user_query: str, file_ids: list, response):
        """response is a dict: {answer, insights, confidence, ...}"""
        key = self._response_key(user_query, file_ids)
        self.data_cache[key] = {
            'kind': 'full_response',
            'response': response,
            'expires': datetime.now() + timedelta(seconds=self.cache_ttl),
            'query': user_query,
            'files': file_ids,
        }

    # ---------- LLM CACHE (semantic key) ----------
    def _llm_cache_key(self, agent: str, user_query: str, file_ids: list) -> str:
        """
        Semantic cache key: same (agent, query, files) ⇒ same key,
        regardless of volatile stuff like conversation context or retry
        error messages. THIS is what fixes the cache-miss-on-repeat bug.
        """
        files_part = ",".join(sorted(file_ids or []))
        key_str = f"{agent}||{user_query.strip().lower()}||{files_part}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_llm_cache(self, agent: str, user_query: str, file_ids: list,
                      prompt_preview: str = "") -> Optional[str]:
        t0 = time.perf_counter()
        key = self._llm_cache_key(agent, user_query, file_ids)
        if key in self.llm_cache:
            entry = self.llm_cache[key]
            if datetime.now() < entry['expires']:
                latency_ms = (time.perf_counter() - t0) * 1000
                record_cache_hit(
                    model=entry.get('model', 'cached'),
                    prompt_preview=prompt_preview or user_query,
                    cached_response=entry['response'],
                    cache_type="llm",
                    latency_ms=latency_ms,
                )
                return entry['response']
            else:
                del self.llm_cache[key]
        return None

    def set_llm_cache(self, agent: str, user_query: str, file_ids: list,
                      response: str, model: str = "") -> None:
        key = self._llm_cache_key(agent, user_query, file_ids)
        self.llm_cache[key] = {
            'response': response,
            'model': model,
            'expires': datetime.now() + timedelta(seconds=self.cache_ttl),
            'created': datetime.now(),
        }

    # ---------- DATA CACHE (already correct, just add logging) ----------
    def get_data_cache(self, query, file_ids):
        t0 = time.perf_counter()
        key = self._hash_data(query, file_ids)
        if key in self.data_cache:
            entry = self.data_cache[key]
            if datetime.now() < entry['expires']:
                latency_ms = (time.perf_counter() - t0) * 1000
                record_cache_hit(
                    model="data-cache",
                    prompt_preview=query,
                    cached_response=str(entry['result'])[:200],
                    cache_type="data",
                    latency_ms=latency_ms,
                )
                return entry['result']
            else:
                del self.data_cache[key]
        return None

    def set_data_cache(self, query, file_ids, result):
        key = self._hash_data(query, file_ids)
        self.data_cache[key] = {
            'result': result,
            'expires': datetime.now() + timedelta(seconds=self.cache_ttl),
            'query': query,
            'files': file_ids,
        }

    # ---------- chat memory + utils (unchanged) ----------
    def add_to_conversation(self, role, content):
        self.conversation.append({'role': role, 'content': content, 'timestamp': datetime.now()})

    def get_conversation_context(self, last_n=4):
        return self.conversation[-last_n:] if self.conversation else []

    def get_conversation_text(self, last_n=4):
        recent = self.get_conversation_context(last_n)
        text = "Previous conversation:\n"
        for msg in recent:
            text += f"{msg['role']}: {msg['content'][:100]}...\n"
        return text if recent else ""

    def _hash_data(self, query, file_ids):
        combined = f"{query}{''.join(sorted(file_ids or []))}"
        return hashlib.md5(combined.encode()).hexdigest()

    def clear_all(self):
        self.llm_cache = {}
        self.data_cache = {}
        self.conversation = []

    def get_stats(self):
        return {
            'llm_cache_size': len(self.llm_cache),
            'data_cache_size': len(self.data_cache),
            'conversation_length': len(self.conversation),
        }


cache_manager = CacheManager()