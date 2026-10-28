"""Minimal OpenAI-compatible /v1/chat/completions server on MUSA.

Does not touch the installed BGE / CrossEncoder environment or experiment results.
Bind to 127.0.0.1 only; NO authentication is provided. Never expose publicly.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ModelBackend:
    def __init__(self, model_dir: str, max_input_tokens: int = 8192):
        import torch
        import torch_musa  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.musa.is_available():
            raise RuntimeError('MUSA 不可用：请在 MTT S4000 实例上运行')
        p = Path(model_dir).expanduser().resolve()
        if not p.is_dir() or not (p / 'config.json').is_file():
            raise FileNotFoundError(f'本地模型未下载完整: {p}')
        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()
        print('Loading tokenizer from', p, flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(p), local_files_only=True, trust_remote_code=False)
        print('Loading FP16 model on CPU...', flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(p), torch_dtype=torch.bfloat16, local_files_only=True,
            trust_remote_code=False, attn_implementation='eager'
        )
        print('Moving model to MUSA...', flush=True)
        self.model.to('musa').eval()
        self.model_name = os.getenv('LOCAL_SERVED_MODEL', 'qwen-local')
        print('Model ready on', torch.musa.get_device_name(0), flush=True)

    def generate(self, messages: list[dict], max_new_tokens: int = 384) -> tuple[str, int, int]:
        messages = [
            {'role': m['role'], 'content': m['content']}
            for m in messages
            if isinstance(m, dict) and m.get('role') in ('system', 'user', 'assistant')
            and isinstance(m.get('content'), str)
        ]
        if not messages:
            raise ValueError('messages 不能为空')
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(text, return_tensors='pt')
        input_tokens = int(tokens['input_ids'].shape[-1])
        if input_tokens > self.max_input_tokens:
            raise ValueError(f'提示词长度 {input_tokens} tokens 超过上限 {self.max_input_tokens}；请减少引用证据或历史轮次')
        gpu_inputs = {k: v.to('musa') for k, v in tokens.items()}
        with self.lock, self.torch.inference_mode():
            generated = self.model.generate(
                **gpu_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = generated[0][input_tokens:]
        result = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if not result:
            raise RuntimeError('本地模型返回了空内容')
        return result, input_tokens, int(new_ids.numel())


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'MHRAG-Local-MUSA/0.1'
        def reply(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            if self.path in ('/health', '/v1/models'):
                if self.path == '/health':
                    return self.reply(200, {'ok': True, 'backend': 'musa', 'model': backend.model_name})
                return self.reply(200, {'object': 'list', 'data': [{'id': backend.model_name, 'object': 'model'}]})
            return self.reply(404, {'error': {'message': 'Not found'}})
        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.reply(404, {'error': {'message': 'Not found'}})
            try:
                n = int(self.headers.get('Content-Length', '-1'))
                if n <= 0 or n > 128 * 1024:
                    raise ValueError('请求长度无效或超过 128KB')
                payload = json.loads(self.rfile.read(n))
                if not isinstance(payload, dict):
                    raise ValueError('请求必须为 JSON 对象')
                if payload.get('stream') is True:
                    raise ValueError('本地服务暂不支持 stream=true')
                name = payload.get('model')
                if name != backend.model_name:
                    raise ValueError(f'模型名称必须为 {backend.model_name}')
                messages = payload.get('messages')
                if not isinstance(messages, list) or len(messages) > 32:
                    raise ValueError('messages 需要是长度不超过 32 的数组')
                max_new = min(max(int(payload.get('max_tokens', 384)), 1), 768)
                answer, prompt_count, completion_count = backend.generate(messages, max_new)
                now = int(time.time())
                return self.reply(200, {
                    'id': f'chatcmpl-local-{uuid.uuid4().hex[:12]}',
                    'object': 'chat.completion', 'created': now, 'model': backend.model_name,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer},
                                 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': prompt_count,
                              'completion_tokens': completion_count,
                              'total_tokens': prompt_count + completion_count}
                })
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                return self.reply(400, {'error': {'message': str(exc)[:350], 'type': 'invalid_request'}})
            except Exception:
                logging.exception('Local generation failed')
                return self.reply(500, {'error': {'message': '本地生成失败，请查看后端日志'}})
    return Handler


def main():
    p = argparse.ArgumentParser(description='Qwen local LLM on MTT S4000 MUSA')
    p.add_argument('--model', default=os.environ.get('LOCAL_LLM_DIR',
        '/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--max-input-tokens', type=int, default=8192)
    args = p.parse_args()
    if args.host not in ('127.0.0.1', 'localhost'):
        p.error('这是无认证的测试服务，只允许绑定 127.0.0.1')
    backend = ModelBackend(args.model, args.max_input_tokens)
    with ThreadingHTTPServer((args.host, args.port), make_handler(backend)) as server:
        print(f'Local LLM ready at http://{args.host}:{args.port}/v1', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
