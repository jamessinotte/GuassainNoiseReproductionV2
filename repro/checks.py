"""Small real-checkpoint preflight before launching a costly sweep."""
import torch
from .config import load_questions
from .io import write_json
from .models import HFModel
from .prompts import build_prompt, conditions


def preflight(config, output):
    if config["model"].get("backend", "hf") != "hf":
        raise ValueError("Use a portable HF config for weight preflight; vLLM requires a separate backend smoke run")
    backend = HFModel(config)
    q = load_questions(config)[0]
    name = conditions(config)[0]
    prepared = backend.prepare(build_prompt(config, q, name))
    baseline = backend.predict(prepared, 42)
    backend.noise.apply(0, 1)
    zero = backend.predict(prepared, 42)
    if baseline != zero:
        raise AssertionError("Zero noise changed prediction")
    sigma = .001
    backend.noise.apply(sigma, 17)
    first = backend.noise.fingerprint()
    backend.noise.apply(sigma, 17)
    if backend.noise.fingerprint() != first:
        raise AssertionError("Seeded perturbation is not deterministic")
    backend.noise.restore()
    backend.noise.assert_restored()
    if backend.predict(prepared, 42) != baseline:
        raise AssertionError("Restoration changed prediction")
    result = {"checks": {"zero_noise_prediction": True, "deterministic_perturbation": True,
                          "bitwise_restoration": True, "restored_prediction": True},
              "baseline_prediction": baseline, "selected_tensor_count": len(backend.noise.parameters),
              "selected_elements": sum(p.numel() for _,_,p in backend.noise.parameters),
              "model": config["model"], "evidence": config["evidence"]}
    write_json(output, result)
    return result
