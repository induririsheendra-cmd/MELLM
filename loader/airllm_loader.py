import time
import logging
import torch
from pathlib import Path
from llama_cpp import Llama

logger = logging.getLogger("LLMRouter.Loader")

# GGUF model registry — maps model_id to (repo_id, filename, [optional_mmproj_filename])
GGUF_REGISTRY = {
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": (
        "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": (
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    ),
    "Qwen/Qwen2.5-Math-1.5B-Instruct": (
        "bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF",
        "Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
    ),
    "BioMistral/BioMistral-7B-DARE-GGUF": (
        "BioMistral/BioMistral-7B-DARE-GGUF",
        "ggml-model-Q2_K.gguf"
    ),
    "AdaptLLM/law-LLM": (
        "mradermacher/magistrate-3.2-3b-it-GGUF",
        "magistrate-3.2-3b-it.Q4_K_M.gguf"
    ),
    "Qwen/Qwen2.5-0.5B-Instruct": (
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ),
}


class ModelLoader:
    def __init__(self, config: dict = None, compression: str = "4bit"):
        self.config = config or {}
        # Merge config registry with default registry
        config_registry = self.config.get("gguf_registry", {})
        self.registry = {**GGUF_REGISTRY, **config_registry}
        
        self.cache: dict = {} # Restored for preloading/unload compatibility
        self.cache_dir = Path.home() / ".cache" / "mellm_gguf"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized ModelLoader with llama-cpp-python (GPU inference)")

    def get_local_models(self) -> dict:
        """Returns a dict of model_id -> availability (bool)."""
        all_models = [self.config.get("router", {}).get("model_id", "")]
        for spec in self.config.get("specialists", {}).values():
            all_models.append(spec.get("model_id", ""))

        result = {}
        for m_id in all_models:
            if not m_id:
                continue
            if m_id in self.registry:
                reg_entry = self.registry[m_id]
                filename = reg_entry[1]
                exists = (self.cache_dir / filename).exists()
                if len(reg_entry) > 2:
                    exists = exists and (self.cache_dir / reg_entry[2]).exists()
                result[m_id] = exists
            else:
                result[m_id] = False
        return result

    def _get_gguf_path(self, model_id: str) -> Path:
        if model_id not in self.registry:
            raise ValueError(f"No GGUF mapping found for model: {model_id}")

        reg_entry = self.registry[model_id]
        repo_id = reg_entry[0]
        filename = reg_entry[1]
        local_path = self.cache_dir / filename

        if not local_path.exists():
            logger.info(f"Downloading GGUF: {repo_id}/{filename} (This may take a while...)")
            self._download_with_progress(repo_id, filename, local_path)
            
        if len(reg_entry) > 2:
            mmproj_filename = reg_entry[2]
            mmproj_path = self.cache_dir / mmproj_filename
            if not mmproj_path.exists():
                logger.info(f"Downloading mmproj: {repo_id}/{mmproj_filename}")
                self._download_with_progress(repo_id, mmproj_filename, mmproj_path)
            return local_path, mmproj_path

        return local_path, None

    def _download_with_progress(self, repo_id: str, filename: str, dest_path: Path) -> None:
        """Downloads a GGUF file from HuggingFace with a rich progress bar."""
        import requests
        from rich.progress import (
            Progress, DownloadColumn, BarColumn,
            TextColumn, TimeRemainingColumn, TransferSpeedColumn
        )
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        import os

        url = hf_hub_url(repo_id=repo_id, filename=filename)
        headers = build_hf_headers(token=os.environ.get("HF_TOKEN"))

        # Stream the download
        response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".tmp")

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task(f"Downloading {filename}", total=total_size)

            try:
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
            except Exception as e:
                # Clean up partial download on failure
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(f"Download failed: {e}")

        # Rename tmp to final only after successful download
        tmp_path.rename(dest_path)
        logger.info(f"Saved to: {dest_path}")

    def stream_download(self, model_id: str):
        """Yields download progress events for the API."""
        if model_id not in self.registry:
            yield {"status": "error", "message": f"Model {model_id} not in registry"}
            return

        reg_entry = self.registry[model_id]
        repo_id = reg_entry[0]
        filename = reg_entry[1]
        dest_path = self.cache_dir / filename
        
        mmproj_filename = reg_entry[2] if len(reg_entry) > 2 else None
        mmproj_dest_path = self.cache_dir / mmproj_filename if mmproj_filename else None

        if dest_path.exists() and (not mmproj_dest_path or mmproj_dest_path.exists()):
            yield {"status": "complete", "message": "File already exists"}
            return

        import requests
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        import os

        url = hf_hub_url(repo_id=repo_id, filename=filename)
        headers = build_hf_headers(token=os.environ.get("HF_TOKEN"))

        try:
            response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
            response.raise_for_status()
        except Exception as e:
            yield {"status": "error", "message": str(e)}
            return

        total_size = int(response.headers.get("content-length", 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(".tmp")

        downloaded = 0
        last_yield_time = time.time()

        yield {"status": "start", "total_size": total_size, "filename": filename}

        try:
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 512): # 512KB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Throttle events to roughly 10Hz to prevent overwhelming the SSE stream
                        current_time = time.time()
                        if current_time - last_yield_time >= 0.1:
                            yield {
                                "status": "progress",
                                "downloaded": downloaded,
                                "total_size": total_size,
                                "percentage": round((downloaded / total_size) * 100, 1) if total_size else 0
                            }
                            last_yield_time = current_time

            tmp_path.rename(dest_path)
            
            # Download mmproj if needed
            if mmproj_filename and mmproj_dest_path and not mmproj_dest_path.exists():
                yield {"status": "progress", "downloaded": 0, "total_size": 1, "percentage": 0, "filename": mmproj_filename}
                mmproj_url = hf_hub_url(repo_id=repo_id, filename=mmproj_filename)
                mm_response = requests.get(mmproj_url, headers=headers, stream=True, allow_redirects=True)
                mm_response.raise_for_status()
                mm_tmp = mmproj_dest_path.with_suffix(".tmp")
                with open(mm_tmp, "wb") as f:
                    for chunk in mm_response.iter_content(chunk_size=1024 * 512):
                        if chunk: f.write(chunk)
                mm_tmp.rename(mmproj_dest_path)

            yield {"status": "complete"}
            logger.info(f"Stream download completed for {model_id}")

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            if 'mm_tmp' in locals() and mm_tmp.exists():
                mm_tmp.unlink()
            yield {"status": "error", "message": f"Download failed: {e}"}

    def delete_model(self, model_id: str) -> bool:
        """Deletes the cached GGUF file for a given model ID."""
        if model_id not in self.registry:
            return False

        _, filename = self.registry[model_id]
        dest_path = self.cache_dir / filename

        # First, ensure it's not currently loaded in VRAM
        self.unload(model_id)

        if dest_path.exists():
            try:
                dest_path.unlink()
                logger.info(f"Deleted cached model file: {dest_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete {dest_path}: {e}")
                return False
        return True # Considered success if it's already gone

    def get(self, model_id: str, is_router: bool = False):
        """
        Loads and returns a Llama model instance.
        Returns (model, None, load_time) to maintain compatibility with orchestrator.
        """
        if model_id in self.cache:
            logger.info(f"Returning cached model: {model_id}")
            model, load_time = self.cache[model_id]
            return model, None, load_time

        logger.info(f"Loading model: {model_id}")
        gguf_path, mmproj_path = self._get_gguf_path(model_id)

        start = time.time()

        # Use a standard 4096 context window for all models to support RAG document injection
        n_ctx = 4096
        
        try:
            chat_handler = None
            if mmproj_path:
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                chat_handler = MoondreamChatHandler(clip_model_path=str(mmproj_path))

            # For specialists (non-router), disable mmap to prevent fragmentation
            # on memory-swapping workflows. Persistent router still uses mmap.
            model = Llama(
                model_path=str(gguf_path),
                n_gpu_layers=-1,   # offload all layers to GPU
                n_ctx=n_ctx,
                n_batch=512,
                use_mmap=(is_router),  # Only mmap the persistent router
                chat_handler=chat_handler,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize Llama model from {gguf_path}: {e}")
            raise RuntimeError(
                f"Model initialization failed. This often happens if the GGUF file is corrupted "
                f"or VRAM is insufficient. Try deleting the file at {gguf_path} and restarting."
            )
        
        load_time = time.time() - start
        logger.info(f"Loaded {model_id} in {load_time:.2f}s")
        
        # Keep track in cache for preloading/orchestration
        self.cache[model_id] = (model, load_time)
        return model, None, load_time

    def unload(self, model_id: str):
        if model_id in self.cache:
            logger.info(f"Unloading model: {model_id}")
            # Use pop to ensure it's removed from cache immediately
            model, _ = self.cache.pop(model_id)
            del model
            
            import gc
            gc.collect()
            
            # Aggressive VRAM release
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # wait for all CUDA ops to complete
                time.sleep(0.2)  # minimal safety buffer
                
            # Extra GC pass to be triple-sure
            gc.collect()
            
            logger.info(f"Cleared VRAM after unloading {model_id}")
