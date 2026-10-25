"""Against the real model: it downloads once on first run, then is cached."""

import math

from llama_index.core import Document, VectorStoreIndex

from llama_index.embeddings.kjarni import KjarniEmbedding


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_meaning_not_words() -> None:
    embedding = KjarniEmbedding()
    query = embedding.get_query_embedding("doctor")
    near, far = embedding.get_text_embedding_batch(["physician", "volcano"])
    assert round(cosine(query, near), 4) == 0.8598
    assert cosine(query, near) > cosine(query, far) + 0.3


def test_single_and_batch_agree() -> None:
    embedding = KjarniEmbedding()
    one = embedding.get_text_embedding("Reykjavik is the capital of Iceland")
    [batch] = embedding.get_text_embedding_batch(["Reykjavik is the capital of Iceland"])
    assert cosine(one, batch) > 0.9999


def test_in_a_vector_store_index() -> None:
    docs = [
        Document(text="Reykjavik is the capital of Iceland."),
        Document(text="The cross-encoder reranks the retrieved passages."),
        Document(text="Sourdough needs a starter fed with flour and water."),
    ]
    index = VectorStoreIndex.from_documents(docs, embed_model=KjarniEmbedding())
    [best] = index.as_retriever(similarity_top_k=1).retrieve("What is Iceland's capital city?")
    assert "Reykjavik" in best.node.get_content()
