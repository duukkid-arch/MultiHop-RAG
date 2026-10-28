"""Download model weights ONLY (never requires a DashScope/OpenAI API key)."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', choices=['1.5b', '7b'], default='1.5b')
    p.add_argument('--root', default='/root/autodl-tmp/models')
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    name = {'1.5b': 'Qwen2.5-1.5B-Instruct', '7b': 'Qwen2.5-7B-Instruct'}[args.size]
    destination = Path(args.root) / name
    destination.mkdir(parents=True, exist_ok=True)
    print('Downloading', name, 'to', destination, flush=True)
    path = snapshot_download(
        repo_id='Qwen/' + name,
        local_dir=str(destination),
        allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', '*.tiktoken', '*.model', '*.txt'],
        max_workers=2,
    )
    print('Download completed:', path)
    print('Weight files:')
    for file in sorted(destination.glob('*.safetensors')):
        print(file.name, file.stat().st_size, 'bytes')

if __name__ == '__main__':
    main()
