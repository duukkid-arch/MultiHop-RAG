#!/usr/bin/env python3
"""Smoke tests with existing local models. Never installs/changes torch."""
import os,time
from pathlib import Path
os.environ.update({'HF_HOME':'/root/autodl-tmp/hf_cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
import torch
import torch_musa
assert torch.musa.is_available(), 'MUSA unavailable'
print('MUSA',torch.__version__,torch.musa.get_device_name(0),flush=True)
x=torch.randn((128,128),device='musa'); y=x@x
print('tensor matmul',tuple(y.shape),float(y[0,0].cpu()),flush=True)
from sentence_transformers import SentenceTransformer,CrossEncoder
bge=Path('/root/autodl-tmp/models/bge-m3')
snaps=Path('/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots')
assert (bge/'modules.json').is_file(),'BGE local model missing'
models=[p for p in snaps.iterdir() if (p/'config.json').is_file() and ((p/'model.safetensors').is_file() or (p/'pytorch_model.bin').is_file())]
assert models,'Reranker local snapshot missing'
print('load BGE...',flush=True)
t=time.monotonic()
e=SentenceTransformer(str(bge),device='musa',trust_remote_code=True)
e.max_seq_length=512
v=e.encode(['What caused this event?','This is a scientific paper.'],batch_size=2,convert_to_numpy=True,normalize_embeddings=True)
print('BGE success',v.shape,'sec',round(time.monotonic()-t,2),flush=True)
del e; torch.musa.empty_cache()
print('load CE...',flush=True)
t=time.monotonic()
r=CrossEncoder(str(models[0]),device='musa',max_length=512,trust_remote_code=False)
s=r.predict([('What caused the event?','The storm caused a power outage.'),('What caused the event?','A study analyzed the new chip.')],batch_size=2,show_progress_bar=False)
print('CE success',s,'sec',round(time.monotonic()-t,2),flush=True)
