import model
import util
import pytest
import torch
from jaxtyping import TypeCheckError

from transformers import GPT2LMHeadModel

ref = GPT2LMHeadModel.from_pretrained("gpt2")


def test_jaxtyping_validation():
    with pytest.raises(TypeCheckError):
        model.LayerNorm(64)(torch.randn(10, 64))  # expects 3D, got 2D


def test_gelu():
    x = torch.randn(5, 5)
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    actual = model.gelu(x)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_layernorm():
    x = torch.randn(64, 64, 768)
    expected = torch.nn.LayerNorm(768)(x)
    actual = model.LayerNorm(768)(x)
    assert torch.allclose(expected, actual, atol=1e-5)


def assert_equal_shapes(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
):
    assert set(expected.keys()) == set(actual.keys())
    for k in expected.keys():
        print(
            f"Checking {k}...expected: {expected[k].shape}, actual: {actual[k].shape}"
        )
        assert expected[k].shape == actual[k].shape


def test_mlp():
    expected: torch.nn.Module = ref.transformer.h[0].mlp  # type: ignore
    actual = model.MultiLayerPerception(768)

    expected_state_dict = util.transpose_state_dict(expected.state_dict())
    actual.load_state_dict(expected_state_dict)

    x = torch.randn(2, 10, 768)
    assert torch.allclose(expected(x), actual(x), atol=1e-5)


def test_casual_self_attention():
    expected: torch.nn.Module = ref.transformer.h[0].attn  # type: ignore
    actual = model.CausalSelfAttention(12, 768, 1024)

    expected_state_dict = util.transpose_state_dict(expected.state_dict())
    actual.load_state_dict(expected_state_dict)

    x = torch.randn(10, 64, 768)
    assert torch.allclose(expected(x)[0], actual(x), atol=1e-5)


def test_transformer_block():
    expected: torch.nn.Module = ref.transformer.h[0]  # type: ignore
    actual = model.TransformerBlock(12, 768, 1024)

    expected_state_dict = util.transpose_state_dict(expected.state_dict())
    actual.load_state_dict(expected_state_dict)

    x = torch.randn(10, 64, 768)
    assert torch.allclose(expected(x)[0], actual(x), atol=1e-4)


def test_transformer():
    expected: torch.nn.Module = ref.transformer  # type: ignore
    actual = model.Transformer()

    expected_state_dict = util.transpose_state_dict(expected.state_dict())
    actual.load_state_dict(expected_state_dict)

    tokenizer = model.Tokenizer()
    token_ids = tokenizer.encode("Hello, my name is")

    x = torch.tensor(token_ids).unsqueeze(0)  # Convert to batch
    assert torch.allclose(expected(x)[0], actual(x), atol=1e-4)


def test_gpt2():
    expected: torch.nn.Module = ref  # type: ignore
    actual = model.GPT2()

    expected_state_dict = util.transpose_state_dict(expected.state_dict())
    actual.load_state_dict(expected_state_dict)

    tokenizer = model.Tokenizer()
    token_ids = tokenizer.encode("Hello, my name is")

    x = torch.tensor(token_ids).unsqueeze(0)  # Convert to batch
    assert torch.allclose(expected(x)[0], actual(x), atol=1e-4)
