from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional
import json
import os


class SemanticCache:
    """Cache queries by semantic similarity"""
    
    def __init__(self, similarity_threshold=0.9, cache_dir='cache'):
        print("🔄 Loading semantic model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
        self.similarity_threshold = similarity_threshold
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache
        self.embeddings = []  # Query embeddings (vectors)
        self.queries = []     # Original query texts
        self.results = []     # Cached results
        
        # Load existing cache from disk
        self._load_cache()
        print(f"✅ Semantic cache ready ({len(self.queries)} cached queries)")
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached result if similar query exists
        
        Returns:
            Cached result if found, None otherwise
        """
        if not self.queries:
            return None
        
        # Convert query to embedding (vector)
        query_embedding = self.model.encode(query)
        
        # Calculate similarity with all cached queries
        similarities = []
        for cached_embedding in self.embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            similarities.append(similarity)
        
        # Find best match
        max_similarity = max(similarities)
        
        if max_similarity >= self.similarity_threshold:
            # Cache HIT
            idx = similarities.index(max_similarity)
            print(f"🎯 Semantic Cache HIT (similarity: {max_similarity:.2%})")
            print(f"   Cached query: {self.queries[idx]}")
            print(f"   Your query:   {query}")
            return self.results[idx]
        
        # Cache MISS
        print(f"❌ Semantic Cache MISS (best match: {max_similarity:.2%})")
        return None
    
    def set(self, query: str, result: str):
        """Store query-result pair in cache"""
        embedding = self.model.encode(query)
        
        self.queries.append(query)
        self.embeddings.append(embedding)
        self.results.append(result)
        
        self._save_cache()
        print(f"💾 Cached: {query[:60]}...")
    
    def clear(self):
        """Clear all cached queries"""
        self.queries = []
        self.embeddings = []
        self.results = []
        self._save_cache()
        print("🗑️  Cache cleared")
    
    def get_stats(self):
        """Get cache statistics"""
        return {
            'total_queries': len(self.queries),
            'threshold': self.similarity_threshold
        }
    
    def _save_cache(self):
        """Save cache to disk"""
        cache_data = {
            'queries': self.queries,
            'embeddings': [emb.tolist() for emb in self.embeddings],  # Convert numpy to list
            'results': self.results
        }
        
        cache_file = f'{self.cache_dir}/semantic_cache.json'
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    def _load_cache(self):
        """Load cache from disk"""
        cache_file = f'{self.cache_dir}/semantic_cache.json'
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            self.queries = cache_data.get('queries', [])
            self.embeddings = [np.array(emb) for emb in cache_data.get('embeddings', [])]
            self.results = cache_data.get('results', [])
            
            print(f"📂 Loaded {len(self.queries)} cached queries from disk")


# Global instance
semantic_cache = SemanticCache()