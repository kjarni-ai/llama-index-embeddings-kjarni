# llama-index-embeddings-kjarni

LlamaIndex embeddings with [Kjarni](https://kjarni.ai): local models in your own process, with no PyTorch, no ONNX Runtime, no GPU and no server.

Kjarni produces the same vectors as sentence-transformers for the same model. The model downloads once on first use and is cached; after that everything works offline.

## Install

```bash
pip install llama-index-embeddings-kjarni
```

## Usage

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.kjarni import KjarniEmbedding

Settings.embed_model = KjarniEmbedding(model_name="minilm-l6-v2")

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
nodes = index.as_retriever().retrieve("What does the contract say about renewal?")
```

| Setting | Default | |
|---|---|---|
| `model_name` | `"minilm-l6-v2"` | Any Kjarni embedding model, e.g. `mpnet-base-v2`, `bge-m3` (multilingual). `kjarni model list` shows all. |
| `device` | `"cpu"` | `"gpu"` runs on Vulkan, Metal or DirectX 12. |
| `embed_batch_size` | `32` | Texts per call into the engine. |
| `query_prefix`, `text_prefix` | `""` | For models trained with prefixes, e.g. `"query: "` and `"passage: "` for E5. |
| `cache_dir` | Kjarni's cache | Where models are stored. |

The model loads on first use, so constructing `KjarniEmbedding` is free.

## License

MIT
# llama-index-embeddings-kjarni
