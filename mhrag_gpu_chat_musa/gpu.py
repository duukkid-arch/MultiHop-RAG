"""Local CUDA / MUSA embeddings and CE reranking with CPU test doubles."""
from __future__ import annotations
import os
import numpy as np

DEFAULT_BGE='/root/autodl-tmp/models/bge-m3'
DEFAULT_CE='/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'

class AcceleratorEncoder:
    def __init__(self,device=None):
        self.embedder=None;self.reranker=None
        self.device=device or os.getenv('CHAT_DEVICE','auto')
        self.bge=os.getenv('BGE_MODEL',DEFAULT_BGE)
        self.ce=os.getenv('CE_MODEL',DEFAULT_CE)
        self.batch=int(os.getenv('CHAT_BATCH_SIZE','4'))
        self.length=int(os.getenv('CHAT_MAX_LENGTH','512'))

    def check(self):
        import torch
        if self.device in ('auto','cuda') and torch.cuda.is_available():
            self.device='cuda'
            return torch.cuda.get_device_name(0)
        if self.device in ('auto','musa'):
            try:
                import torch_musa  # noqa: F401
                if torch.musa.is_available():
                    self.device='musa'
                    return torch.musa.get_device_name(0)
            except (ImportError, RuntimeError, OSError):
                pass
        if self.device=='cpu':return 'CPU (development only)'
        raise RuntimeError(f'{self.device} 不可用；4090 请安装兼容 CUDA 的 PyTorch 并设 CHAT_DEVICE=cuda')

    def embed(self,texts):
        if not texts:return np.empty((0,1024),dtype=np.float32)
        if self.embedder is None:
            self.check()
            from sentence_transformers import SentenceTransformer
            self.embedder=SentenceTransformer(self.bge,device=self.device,trust_remote_code=True)
            self.embedder.max_seq_length=self.length
        rows=self.embedder.encode(list(texts),batch_size=self.batch,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        mat=np.asarray(rows,dtype=np.float32)
        if mat.ndim!=2 or mat.shape[1]!=1024:raise RuntimeError(f'Expected BGE-M3 1024 dims, got {mat.shape}')
        return mat

    def score(self,pairs):
        if not pairs:return []
        if self.reranker is None:
            self.check()
            from sentence_transformers import CrossEncoder
            self.reranker=CrossEncoder(self.ce,device=self.device,max_length=self.length,trust_remote_code=False)
        vals=self.reranker.predict(pairs,batch_size=self.batch,show_progress_bar=False)
        return [float(x) for x in vals]


# Keep old imports and old MUSA scripts working.
MUSAEncoder=AcceleratorEncoder
