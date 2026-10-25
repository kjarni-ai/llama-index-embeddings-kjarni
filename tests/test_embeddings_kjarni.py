"""Behaviour that does not need a real model: a fake stands in for Kjarni."""

import numpy as np
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import ValidationError

from llama_index.embeddings.kjarni import KjarniEmbedding


class FakeEmbedder:
    def __init__(self, as_numpy: bool) -> None:
        self.as_numpy = as_numpy
        self.batches: list = []
        self.singles: list = []

    def encode(self, text):
        self.singles.append(text)
        return [float(len(text)), 1.0]

    def encode_batch(self, texts):
        self.batches.append(list(texts))
        rows = [[float(len(t)), 1.0] for t in texts]
        return np.array(rows, dtype=np.float32) if self.as_numpy else rows


def with_fake(as_numpy: bool = True, **kwargs):
    embedding = KjarniEmbedding(**kwargs)
    fake = FakeEmbedder(as_numpy)
    embedding._client = fake
    return embedding, fake


def test_class() -> None:
    assert BaseEmbedding.__name__ in [b.__name__ for b in KjarniEmbedding.__mro__]
    assert KjarniEmbedding.class_name() == "KjarniEmbedding"


def test_construction_loads_nothing() -> None:
    assert KjarniEmbedding(model_name="no-such-model")._client is None


@pytest.mark.parametrize("as_numpy", [True, False])
def test_batches_come_back_as_plain_float_lists(as_numpy: bool) -> None:
    embedding, _ = with_fake(as_numpy)
    vectors = embedding.get_text_embedding_batch(["a", "bbb"])
    assert vectors == [[1.0, 1.0], [3.0, 1.0]]
    assert all(type(x) is float for row in vectors for x in row)


def test_batch_size_and_prefixes() -> None:
    embedding, fake = with_fake(embed_batch_size=2, query_prefix="query: ", text_prefix="passage: ")
    embedding.get_text_embedding_batch(["a", "b", "c"])
    assert fake.batches == [["passage: a", "passage: b"], ["passage: c"]]
    embedding.get_query_embedding("q")
    assert fake.singles == ["query: q"]


async def test_async_paths() -> None:
    embedding, _ = with_fake()
    assert await embedding.aget_query_embedding("ab") == [2.0, 1.0]
    assert await embedding.aget_text_embedding_batch(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]


def test_bad_settings_are_rejected() -> None:
    with pytest.raises(ValidationError):
        KjarniEmbedding(embed_batch_size=0)
    with pytest.raises(ValidationError):
        KjarniEmbedding(device="tpu")
