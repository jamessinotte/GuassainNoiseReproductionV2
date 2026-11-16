import copy
import pytest
import torch
from repro.noise import WeightNoise, factor_std, tensor_hash


def small_model(dtype=torch.float32):
    torch.manual_seed(123)
    return torch.nn.Sequential(torch.nn.Embedding(32,8), torch.nn.LayerNorm(8), torch.nn.Linear(8,4)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_zero_noise_and_repeated_restore_are_bit_exact(dtype):
    model = small_model(dtype)
    noise = WeightNoise(model)
    clean = {n:tensor_hash(p) for n,p in model.named_parameters()}
    for seed in range(5):
        noise.apply(.003, seed)
        noise.apply(0, seed)
        noise.assert_restored()
    assert clean == {n:tensor_hash(p) for n,p in model.named_parameters()}


def test_draws_match_source_and_do_not_touch_global_rng():
    model = small_model()
    noise = WeightNoise(model)
    state = torch.random.get_rng_state().clone()
    noise.apply(.002, 17)
    assert torch.equal(state, torch.random.get_rng_state())
    for index, name, parameter in noise.parameters:
        generator = torch.Generator().manual_seed(17+index)
        expected = noise.originals[name] + torch.normal(0., .002, size=parameter.shape,
                                                       generator=generator, dtype=parameter.dtype)
        assert torch.equal(parameter, expected)
    first = noise.fingerprint()
    torch.rand(2000)
    noise.apply(.002, 17)
    assert noise.fingerprint() == first
    noise.apply(.002,18)
    assert noise.fingerprint() != first


def test_scale_is_absolute_and_same_direction_across_magnitudes():
    model = torch.nn.Linear(512,512,bias=False)
    with torch.no_grad():
        model.weight.fill_(100.)
    noise = WeightNoise(model)
    noise.apply(.02, 29)
    first = model.weight.detach().clone() - 100
    assert abs(float(first.std())-.02) < .0002
    noise.apply(.04,29)
    second = model.weight.detach().clone() - 100
    assert torch.allclose(second, first*2, atol=1e-5)


def test_coverage_includes_embeddings_norms_and_head_and_restores_on_error():
    model = small_model()
    noise = WeightNoise(model)
    assert len(noise.layout()) == len(list(model.named_parameters()))
    with pytest.raises(RuntimeError):
        with noise.trial(.1,12):
            assert all(not torch.equal(p, noise.originals[n]) for _,n,p in noise.parameters)
            raise RuntimeError("evaluation failed")
    noise.assert_restored()


def test_source_lora_is_noise_plus_nonzero_initialized_a_not_pure_ba_noise():
    from transformers import LlamaConfig, LlamaForCausalLM
    from repro.models import attach_noise_adapter
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    config = {"noise": {"rank": 8, "initialization_seed": 42}}
    model = attach_noise_adapter(model, config)
    base_before = {n:tensor_hash(p) for n,p in model.named_parameters() if "lora_" not in n}
    noise = WeightNoise(model,"lora","factor_std",8)
    assert all("lora_A" in n or "lora_B" in n for _,n,_ in noise.parameters)
    assert all(bool(torch.any(p != 0)) for _,n,p in noise.parameters if "lora_A" in n)
    assert all(bool(torch.all(p == 0)) for _,n,p in noise.parameters if "lora_B" in n)
    generator = torch.Generator().manual_seed(9)
    expected = {n:noise.originals[n] + torch.normal(0., .1, size=p.shape, dtype=p.dtype, generator=generator)
                for _,n,p in noise.parameters}
    noise.apply(.1,9)
    assert all(torch.equal(p,expected[n]) for _,n,p in noise.parameters)
    assert base_before == {n:tensor_hash(p) for n,p in model.named_parameters() if "lora_" not in n}
    noise.restore()
    noise.assert_restored()
    assert factor_std(.001,"paper_direct_std",8) == pytest.approx(8**.25 * .001**.5)

