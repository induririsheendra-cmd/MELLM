import base64
import yaml
from loader.airllm_loader import ModelLoader

with open('user_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

loader = ModelLoader(config)
model, _, _ = loader.get('moondream/moondream2-gguf')

# Use the second uploaded image from artifacts
img_path = r"C:\Users\Rishi\.gemini\antigravity-ide\brain\688245e8-1c99-419e-90f8-539d9ac8be34\.user_uploaded\media_1788979360219.png"
# Map path to WSL
img_path = "/mnt/c/Users/Rishi/.gemini/antigravity-ide/brain/688245e8-1c99-419e-90f8-539d9ac8be34/.user_uploaded/media_1788979360219.png"
with open(img_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

messages = [
    {
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
            {'type': 'text', 'text': 'Extract all the text from this image exactly as written. Be very detailed.'}
        ]
    }
]

res = model.create_chat_completion(messages=messages, max_tokens=1024, temperature=0.1)
print(res['choices'][0]['message']['content'])
