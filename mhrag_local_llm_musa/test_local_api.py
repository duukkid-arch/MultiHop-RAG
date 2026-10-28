"""Calls the model with no API credentials; checks local server connectivity."""
import json
import urllib.request

payload={'model':'qwen-local','messages':[{'role':'user','content':'请用一句话解释什么是系统综述。'}], 'max_tokens':100}
req=urllib.request.Request('http://127.0.0.1:8001/v1/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'},method='POST')
with urllib.request.urlopen(req,timeout=180) as resp:
    obj=json.load(resp)
print('本地回答:',obj['choices'][0]['message']['content'])
print('Token 用量:',obj.get('usage'))
