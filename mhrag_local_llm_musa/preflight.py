"""Non-mutating environment check for the exact MUSA runtime."""
import importlib.metadata
import importlib.util
import os
from pathlib import Path

for name in ('torch','torch_musa','transformers','tokenizers','huggingface_hub','safetensors'):
    try:
        ver=importlib.metadata.version(name.replace('_','-'))
    except importlib.metadata.PackageNotFoundError:
        ver='NOT INSTALLED'
    print(f'{name:20} {ver}')
try:
    import torch
    import torch_musa
    print('MUSA available:', torch.musa.is_available())
    if torch.musa.is_available():
        print('GPU:',torch.musa.get_device_name(0))
except Exception as e:
    print('MUSA check error:',type(e).__name__, str(e)[:250])
p=Path(os.environ.get('LOCAL_LLM_DIR','/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct'))
print('Model:',p,'exists:',(p/'config.json').exists())
print('Weights:',[(q.name,q.stat().st_size) for q in p.glob('*.safetensors')])
