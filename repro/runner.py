"""Readable paired experiment loop with complete per-question evidence."""
import copy
import os
from datetime import datetime, timezone
from pathlib import Path
import torch
from .config import load_questions, magnitudes, validate
from .io import ROOT, code_identity, digest, environment, json_digest, read_json, write_json
from .models import HFModel, VLLMModel
from .prompts import build_prompt, conditions
from .scoring import score


def run(config, output):
    validate(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config)
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    questions = load_questions(config)
    identity = {"config": config, "code": code_identity(), "models": read_json(ROOT / "assets/models.lock.json"),
                "sources": read_json(ROOT / "assets/sources.lock.json"),
                "data_sha256": digest((ROOT / config["data"]["path"]).read_bytes()), "environment": environment()}
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if read_json(metadata_path)["identity"] != identity:
            raise ValueError("Resume identity changed (code/config/data/models/environment). Use a new output directory.")
    else:
        write_json(metadata_path, {"identity": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
                                   "status": "started", "interpretation": config["evidence"]})
    try:
        backend = VLLMModel(config, output) if config["model"].get("backend", "hf") == "vllm" else HFModel(config)
        write_json(output / "effective_model.json", backend.details)
        write_json(output / "parameter_layout.json", backend.noise.layout())
        names = conditions(config)
        prepared = {name: [backend.prepare(build_prompt(config, q, name)) for q in questions] for name in names}
        write_json(output / "inputs.json", {name: [{"question_id": i, "target": q["answer"], **p}
                                                  for i, (q, p) in enumerate(zip(questions, prepared[name]))]
                                             for name in names})
        zero_predictions = None
        count = 0
        for measurement, seeds in config["seed_groups"].items():
            for magnitude_index, magnitude in enumerate(magnitudes(config)):
                for seed in seeds:
                    cell = f"{measurement}_level{magnitude_index:03d}_seed{seed}"
                    checkpoint = output / "trials" / (cell + ".json")
                    if checkpoint.exists():
                        prior = read_json(checkpoint)
                        if prior["checksum"] != json_digest(prior["trial"]):
                            raise ValueError(f"Checkpoint checksum failed: {checkpoint}")
                        if magnitude == 0:
                            zero_predictions = prior["trial"]["predictions"]
                        continue
                    predictions = []
                    reused_zero = magnitude == 0 and zero_predictions is not None and config["evaluation"]["method"] == "next_token_exact"
                    # One perturbation, all questions, all conditions, then reset.
                    with backend.noise.trial(magnitude, seed):
                        if isinstance(backend, VLLMModel):
                            backend.set_adapter(magnitude)
                        weight_hash = backend.noise.fingerprint() if config.get("hash_each_trial", False) else None
                        if reused_zero:
                            predictions = copy.deepcopy(zero_predictions)
                        else:
                            for name in names:
                                for i, question in enumerate(questions):
                                    # Separate from noise RNG, and shared across paired
                                    # conditions. No seed is recovered where unreported.
                                    generation_seed = config["evaluation"].get("generation_seed", 42) + i
                                    result = backend.predict(prepared[name][i], generation_seed)
                                    answer, correct = score(result["output"], question["answer"],
                                                            config["evaluation"]["method"], result.get("full_text"))
                                    prediction = {"question_id": i, "condition": name, "target": question["answer"],
                                                  "answer": answer, "correct": bool(correct),
                                                  "generation_seed": generation_seed, **result}
                                    method = config["evaluation"]["method"]
                                    if method in {"inspect_numeric_end", "legacy_cot", "next_token_exact"}:
                                        corrected = {"inspect_numeric_end": "numeric_exact_corrected", "legacy_cot": "continuation_number_corrected",
                                                     "next_token_exact": "next_token_numeric_key_corrected"}[method]
                                        fixed_answer, fixed_correct = score(result["output"], question["answer"], corrected)
                                        prediction["corrected_diagnostic"] = {"method": corrected, "answer": fixed_answer,
                                                                               "correct": bool(fixed_correct)}
                                    predictions.append(prediction)
                            if magnitude == 0 and config["evaluation"]["method"] == "next_token_exact":
                                zero_predictions = copy.deepcopy(predictions)
                        if weight_hash is not None and backend.noise.fingerprint() != weight_hash:
                            raise AssertionError("Evaluation changed the paired weights")
                    if config.get("verify_every_restoration", False):
                        backend.noise.assert_restored()
                    trial = {"cell": cell, "measurement": measurement, "magnitude": magnitude, "seed": seed,
                             "noise_kind": config["noise"]["kind"], "noise_units": config["noise"]["units"],
                             "weight_sha256": weight_hash, "paired_conditions": names, "predictions": predictions,
                             "zero_baseline_reused": reused_zero,
                             "accuracy": {name: sum(p["correct"] for p in predictions if p["condition"] == name) / len(questions)
                                          for name in names}}
                    write_json(checkpoint, {"trial": trial, "checksum": json_digest(trial)})
                    count += 1
                    print(f"{cell} sigma={magnitude:g} {trial['accuracy']}", flush=True)
        backend.noise.assert_restored()
        write_json(output / "completion.json", {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                                               "trials": sum(len(s) for s in config["seed_groups"].values()) * len(magnitudes(config)),
                                               "restoration": "bitwise verified"})
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                            "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise
