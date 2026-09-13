from .base_specialist import BaseSpecialist
from typing import Generator

class VisionSpecialist(BaseSpecialist):
    def _get_temperature(self) -> float:
        return 0.5

    def _build_messages(self, prompt: str, image_data: str = None) -> list:
        if image_data:
            import base64
            import io
            from PIL import Image

            try:
                # Strip prefix if present
                if "," in image_data:
                    b64_str = image_data.split(",", 1)[1]
                else:
                    b64_str = image_data

                # Decode and open image
                img_bytes = base64.b64decode(b64_str)
                img = Image.open(io.BytesIO(img_bytes))

                # Convert to RGB (handles PNG transparency)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Resize if too large to prevent bitmap allocation errors
                max_size = 768
                if max(img.width, img.height) > max_size:
                    ratio = max_size / max(img.width, img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Save to JPEG buffer
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                new_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                url = f"data:image/jpeg;base64,{new_b64}"
            except Exception as e:
                print(f"Error processing image for vision model: {e}")
                # Fallback to original
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
