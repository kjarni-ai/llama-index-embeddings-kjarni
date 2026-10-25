"""Kjarni embeddings for LlamaIndex."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, List, Literal, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr


class KjarniEmbedding(BaseEmbedding):
    """Kjarni embedding models, running locally in your own process.

    Kjarni is a native inference engine with Python bindings. It produces the
    same vectors as sentence-transformers for the same model, without PyTorch,
    ONNX Runtime, a GPU or a server. The model downloads once on first use and
    is cached; after that everything works offline.

    Examples:
        ``pip install llama-index-embeddings-kjarni``

        .. code-block:: python

            from llama_index.core import Settings
            from llama_index.embeddings.kjarni import KjarniEmbedding

            Settings.embed_model = KjarniEmbedding(model_name="minilm-l6-v2")
    """

    model_name: str = Field(default="minilm-l6-v2", description="Kjarni model name, e.g. minilm-l6-v2, mpnet-base-v2 or bge-m3.")
    embed_batch_size: int = Field(default=32, gt=0, le=2048, description="Texts per call into the engine.")
    device: Literal["cpu", "gpu"] = Field(default="cpu", description="Where to run the model. The GPU path uses Vulkan, Metal or DirectX 12.")
    normalize: bool = Field(default=True, description="L2-normalize vectors, so a dot product is the cosine similarity.")
    query_prefix: str = Field(default="", description="Prepended to every query, for models trained with one, e.g. 'query: ' for E5.")
    text_prefix: str = Field(default="", description="Prepended to every text, for models trained with one, e.g. 'passage: ' for E5.")
    cache_dir: Optional[str] = Field(default=None, description="Where models are cached. Defaults to Kjarni's own cache directory.")
    quiet: bool = Field(default=True, description="Suppress Kjarni's progress output, including download progress.")

    # Loaded on first use rather than at construction, so creating the object
    # is free and never touches the network.
    _client: Any = PrivateAttr(default=None)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @classmethod
    def class_name(cls) -> str:
        return "KjarniEmbedding"

    def _embedder(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from kjarni import Embedder

                    self._client = Embedder(
                        model=self.model_name,
                        device=self.device,
                        cache_dir=self.cache_dir,
                        normalize=self.normalize,
                        quiet=self.quiet,
                    )
        return self._client

    def _get_query_embedding(self, query: str) -> List[float]:
        return [float(x) for x in self._embedder().encode(self.query_prefix + query)]

    def _get_text_embedding(self, text: str) -> List[float]:
        return [float(x) for x in self._embedder().encode(self.text_prefix + text)]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return _as_lists(self._embedder().encode_batch([self.text_prefix + t for t in texts]))

    # The engine runs on native threads; the async forms hand the work to a
    # worker so an event loop is never blocked while a batch is embedded.
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self._get_text_embeddings, texts)


def _as_lists(vectors: Any) -> List[List[float]]:
    """Kjarni returns a NumPy array when NumPy is installed and lists when it
    is not; LlamaIndex wants plain lists of floats either way."""
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return [[float(x) for x in row] for row in vectors]
