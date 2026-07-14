from __future__ import annotations

import torch

from esa.generation import parse_engine_spec
from esa.model import ESAModel, ESAModelConfig


class TinyTokenizer:
    def encode(self, text: str):
        return [ord(ch) % 32 for ch in text]

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


def make_model(
    device: str = "cpu",
):
    return ESAModel(
        ESAModelConfig(
            vocab_size=32,
            block=32,
            n_layer=2,
            head=4,
            embd=32,
            dropout=0.0,
            precision="fp32",
        ),
        device=device,
    )


def test_optimized_model_defaults():
    cfg = ESAModelConfig(vocab_size=32)
    assert cfg.backend == "thunder"
    assert cfg.compass is None
    assert cfg.training_compile is True
    model = ESAModel(cfg)
    assert model.blocks[0].esa.compass == 16
    assert cfg.training_compile_mode == "reduce-overhead"


def test_engine_parser():
    spec = parse_engine_spec("thunder_compiled_16")
    assert spec.backend == "thunder"
    assert spec.compiled is True
    assert spec.compass == 16
    assert parse_engine_spec("flare_compiled").compiled is True
    assert parse_engine_spec("lightning").backend == "lightning"


def test_prefill_state_equivalence_cpu():
    torch.manual_seed(1)

    model = make_model(
        device="cpu",
    )

    ids = torch.randint(
        0,
        32,
        (1, 9),
    )

    _, thunder_state, _ = model.prefill(
        ids,
        engine="thunder_16",
    )

    _, lightning_state, _ = model.prefill(
        ids,
        engine="lightning",
    )

    torch.testing.assert_close(
        thunder_state,
        lightning_state,
    )


def test_seek_and_backward_compatibility():
    torch.manual_seed(1)
    model = make_model()
    ids = torch.randint(0, 32, (1, 4))
    a = model.generate_ids(ids, seek=2, compile=False, temperature=0.0)
    b = model.generate(
        input_ids=ids,
        max_new_tokens=2,
        compile=False,
        temperature=0.0,
    )
    assert a.shape == b.shape == (1, 6)


def test_raw_text_generate_api():
    model = make_model()
    text = model.generate(
        "hi",
        tokenizer=TinyTokenizer(),
        seek=1,
        compile=False,
        temperature=0.0,
    )
    assert isinstance(text, str)
