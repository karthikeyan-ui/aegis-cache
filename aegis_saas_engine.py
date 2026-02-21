import numpy as np
import hashlib
import pickle
import os
import threading
from aegis_core_math import AegisAlgorithmicEngine, RBAC_KDTree # Your existing math

class AegisSaaSEngine(AegisAlgorithmicEngine):
    def __init__(self, alpha=0.7, beta=0.3, storage_path="./aegis_data/"):
        super().__init__(alpha, beta)
        self.storage_path = storage_path
        self.lock = threading.RLock() # Thread safety for concurrent API requests
        
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            
        self.load_snapshot()

    def insert_to_cache(self, tenant_id: str, role_id: str, prompt: str, response: str):
        """Thread-safe insertion"""
        with self.lock:
            super().insert_to_cache(tenant_id, role_id, prompt, response)
            # In production, you would trigger an async disk write here 
            # instead of blocking, but this ensures safety.

    def save_snapshot(self):
        """Serializes the KD-Trees into a secure binary file."""
        with self.lock:
            file_path = os.path.join(self.storage_path, "vault_snapshot.aegis")
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(self.memory_vault, f)
                print(f"💾 AegisCache State Saved: {len(self.memory_vault)} partitions backed up.")
            except Exception as e:
                print(f"CRITICAL ERROR Saving Snapshot: {e}")

    def load_snapshot(self):
        """Reloads the mathematical state if the SaaS server restarts."""
        file_path = os.path.join(self.storage_path, "vault_snapshot.aegis")
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    self.memory_vault = pickle.load(f)
                print(f"🔄 AegisCache Restored from Disk: {len(self.memory_vault)} partitions active.")
            except Exception as e:
                print(f"CRITICAL ERROR Loading Snapshot: {e}")
                self.memory_vault = {}
                