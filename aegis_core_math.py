import numpy as np
import hashlib
import math
from collections import Counter

# ==========================================
# 1. THE KD-TREE DATA STRUCTURE (Indexer)
# ==========================================

class KDNode:
    """A node in the K-Dimensional Tree."""
    def __init__(self, vector, payload, left=None, right=None, axis=0):
        self.vector = vector
        self.payload = payload
        self.left = left
        self.right = right
        self.axis = axis

class RBAC_KDTree:
    """A custom KD-Tree optimized for fast semantic vector searches."""
    def __init__(self, k_dimensions=128):
        self.root = None
        self.k = k_dimensions

    def insert(self, vector: np.ndarray, payload: dict):
        if self.root is None:
            self.root = KDNode(vector, payload, axis=0)
            return

        node = self.root
        while True:
            axis = node.axis
            next_axis = (axis + 1) % self.k
            
            if vector[axis] < node.vector[axis]:
                if node.left is None:
                    node.left = KDNode(vector, payload, axis=next_axis)
                    break
                node = node.left
            else:
                if node.right is None:
                    node.right = KDNode(vector, payload, axis=next_axis)
                    break
                node = node.right

    def nearest_neighbor(self, target_vector: np.ndarray, threshold_distance: float):
        best_node = None
        best_dist = float('inf')

        def _search(node):
            nonlocal best_node, best_dist
            if node is None:
                return

            dist = np.sum((node.vector - target_vector) ** 2)
            
            if dist < best_dist:
                best_dist = dist
                best_node = node

            axis = node.axis
            diff = target_vector[axis] - node.vector[axis]

            close_branch = node.left if diff < 0 else node.right
            far_branch = node.right if diff < 0 else node.left

            _search(close_branch)

            if (diff ** 2) < best_dist:
                _search(far_branch)

        _search(self.root)
        
        if best_dist <= threshold_distance:
            return best_node
        return None

# ==========================================
# 2. THE MAIN ALGORITHMIC ENGINE
# ==========================================

class AegisAlgorithmicEngine:
    def __init__(self, alpha=0.7, beta=0.3):
        self.alpha = alpha
        self.beta = beta
        self.memory_vault = {} 

    def _generate_rbac_hash(self, tenant_id: str, role_id: str) -> str:
        salt = "aegis_enterprise_v1"
        raw_key = f"{tenant_id}_{role_id}_{salt}"
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def _compute_bm25(self, query_tokens: list, doc_tokens: list, k1=1.5, b=0.75) -> float:
        doc_len = len(doc_tokens)
        avg_doc_len = 10 
        score = 0.0
        doc_counts = Counter(doc_tokens)
        
        for term in query_tokens:
            if term in doc_counts:
                freq = doc_counts[term]
                idf = math.log(2.0) 
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += idf * (numerator / denominator)
                
        return min(score / 5.0, 1.0) 

    def _tokenize_and_vectorize(self, text: str) -> tuple:
        tokens = text.lower().replace("?", "").replace(".", "").split()
        vector = np.zeros(128)
        for t in tokens:
            idx = int(hashlib.md5(t.encode()).hexdigest(), 16) % 128
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return tokens, vector

    def insert_to_cache(self, tenant_id: str, role_id: str, prompt: str, response: str):
        rbac_hash = self._generate_rbac_hash(tenant_id, role_id)
        tokens, vector = self._tokenize_and_vectorize(prompt)
        
        if rbac_hash not in self.memory_vault:
            self.memory_vault[rbac_hash] = RBAC_KDTree(k_dimensions=128)
            
        payload = {"tokens": tokens, "response": response, "prompt": prompt}
        self.memory_vault[rbac_hash].insert(vector, payload)

    def search_cache(self, tenant_id: str, role_id: str, prompt: str, semantic_threshold=0.85):
        rbac_hash = self._generate_rbac_hash(tenant_id, role_id)
        
        if rbac_hash not in self.memory_vault:
            return None, 0.0
            
        query_tokens, query_vector = self._tokenize_and_vectorize(prompt)
        euclidean_threshold = 2.0 * (1.0 - semantic_threshold)
        
        best_node = self.memory_vault[rbac_hash].nearest_neighbor(query_vector, euclidean_threshold)
        
        if best_node is None:
            return None, 0.0
            
        lexical_score = self._compute_bm25(query_tokens, best_node.payload["tokens"])
        semantic_score = 1.0 - (np.sum((best_node.vector - query_vector)**2) / 2.0)
        final_score = (self.alpha * semantic_score) + (self.beta * lexical_score)
        
        if final_score >= semantic_threshold:
            return best_node.payload["response"], final_score
            
        return None, final_score
        