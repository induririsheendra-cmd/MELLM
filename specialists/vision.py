from .base_specialist import BaseSpecialist
from typing import Generator

class VisionSpecialist(BaseSpecialist):
    def _get_temperature(self) -> float:
        return 0.5

    def _build_messages(self, prompt: str, image_data: str = None) -> list:
        if image_data:
            if image_data.startswith("data:image"):
                url = image_data
            else:
                url = f"data:image/jpeg;base64,{image_data}"
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        return [{"role": "user", "content": prompt}]

    def generate(self, prompt: str, image_data: str = None) -> str:
        messages = self._build_messages(prompt, image_data)
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self._get_temperature(),
        )
        return self._postprocess(
            response["choices"][0]["message"]["content"]
        )

    def stream_generate(self, prompt: str, image_data: str = None) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, image_data)
        stream = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self._get_temperature(),
            stream=True
        )
        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token
