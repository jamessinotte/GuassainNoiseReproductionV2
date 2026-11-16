"""JSON configurations with a single, visible inheritance mechanism."""
import copy
import math
from pathlib import Path
from .io import ROOT, read_json


def merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError("Configuration inheritance cycle")
    config = read_json(path)
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, ancestors + (path,)), config)
    return config


def magnitudes(config):
    sweep = config["noise"]["sweep"]
    if isinstance(sweep, list):
        values = sweep
    else:
        count = sweep["count"]
        if count < 2:
            raise ValueError("A linspace sweep needs at least two points")
        values = [sweep["start"] + (sweep["stop"] - sweep["start"]) * i / (count - 1)
                  for i in range(count)]
    if not values or len(set(values)) != len(values) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Noise magnitudes must be finite, distinct, nonnegative")
    return values


def validate(config):
    required = {"name", "evidence", "model", "data", "prompt", "noise", "seed_groups", "evaluation"}
    if required - config.keys():
        raise ValueError(f"Missing config fields: {sorted(required - config.keys())}")
    if config["noise"]["kind"] not in {"direct", "lora"}:
        raise ValueError("Only direct Gaussian noise and source LoRA Gaussian noise are implemented")
    if config["noise"].get("mean", 0) != 0 or config["noise"].get("percentage", 1) != 1:
        raise ValueError("These protocols require zero mean and 100% of selected parameters")
    if config["noise"]["kind"] == "lora" and config["noise"]["units"] not in {"factor_std", "paper_direct_std"}:
        raise ValueError("LoRA magnitudes must explicitly state their units")
    if config["noise"]["kind"] == "direct" and config["noise"]["units"] != "absolute_weight_std":
        raise ValueError("Direct noise uses absolute weight-space standard deviation")
    if config["model"].get("quantization"):
        raise ValueError("Quantized weights are not a faithful dense floating-point perturbation")
    if config["model"].get("backend", "hf") == "vllm" and config["noise"]["kind"] != "lora":
        raise ValueError("vLLM is supported only for source LoRA noise")
    seen = set()
    for measurement, seeds in config["seed_groups"].items():
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f"Invalid seeds in {measurement}")
        if seen.intersection(seeds):
            raise ValueError("Repeated noise seeds across measurements are not independent measurements")
        seen.update(seeds)
    values = magnitudes(config)
    if 0 not in values and not config.get("scaling_experiment"):
        raise ValueError("A sweep must include a true zero-noise baseline")
    return config


def load_questions(config):
    spec = config["data"]
    path = ROOT / spec["path"]
    from .io import digest
    if spec.get("sha256") and digest(path.read_bytes()) != spec["sha256"]:
        raise ValueError(f"Dataset hash mismatch: {path}")
    questions = [__import__("json").loads(line) for line in path.read_text().splitlines() if line.strip()]
    if spec.get("expected_count") is not None and len(questions) != spec["expected_count"]:
        raise ValueError("Dataset row count differs from source manifest")
    if spec.get("limit") is not None:
        questions = questions[:spec["limit"]]
    if not questions or any("answer" not in q or q["answer"] is None for q in questions):
        raise ValueError("Need nonempty questions with answer keys; preserve original key types")
    return questions

