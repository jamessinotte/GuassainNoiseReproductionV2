"""Checkpoint loading and the two concrete inference paths used here."""
import gc
import json
from pathlib import Path
import torch
from .io import ROOT, digest, read_json
from .noise import WeightNoise
from .prompts import encode_prompt, format_text


def revision(model_id):
    lock = read_json(ROOT / "assets/models.lock.json")
    entry = lock.get(model_id)
    if not entry or not entry.get("revision"):
        raise ValueError(f"No immutable revision for {model_id}. Run scripts/prepare.py --lock-models")
    return entry["revision"]


def load_checkpoint(config, device):
    import importlib.util
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = config["model"]
    if spec.get("attention") == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise ImportError("This preset requires flash-attn. See requirements-phi-small.txt and docs/EXPERIMENTS.md before launching the sweep.")
    model_id = spec["id"]
    tokenizer_id = spec.get("tokenizer_id") or model_id
    dtype = getattr(torch, spec["dtype"])
    common = {"trust_remote_code": spec.get("trust_remote_code", False), "cache_dir": str(ROOT / "models/hf")}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision(tokenizer_id), **common)
    kwargs = dict(revision=revision(model_id), torch_dtype=dtype, low_cpu_mem_usage=True, **common)
    if spec.get("attention"):
        kwargs["attn_implementation"] = spec["attention"]
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if spec.get("adapter_id"):
        from peft import PeftModel
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        # The source does not seed the added embedding row. This explicit choice
        # is recorded as a reconstruction decision, not a recovered training seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec["resize_seed"])
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, spec["adapter_id"],
                                         revision=revision(spec["adapter_id"]), cache_dir=str(ROOT / "models/hf")).merge_and_unload()
    model.eval()
    model.to(device)
    return model, tokenizer


def attach_noise_adapter(model, config):
    from peft import LoraConfig, get_peft_model
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not any(x in name for x in ["lm_head", "head", "output"]):
            if name.split(".")[-1] not in targets:
                targets.append(name.split(".")[-1])
    if not targets:
        raise ValueError("No source-compatible linear modules found")
    rank = config["noise"]["rank"]
    # Upstream seeds AFTER get_peft_model, leaving A_0 initialization unspecified.
    # Seed it separately so replay does not depend on unrelated RNG operations.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["noise"]["initialization_seed"])
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank, target_modules=targets,
                                               lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    model.eval()
    return model


class HFModel:
    def __init__(self, config):
        self.config = config
        device = config["model"]["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Full presets require a GPU; use configs/smoke.json for CPU validation.")
        # Source LoRA factors are initialized on a temporary CPU model.
        model, self.tokenizer = load_checkpoint(config, "cpu" if config["noise"]["kind"] == "lora" else device)
        if config["noise"]["kind"] == "lora":
            model = attach_noise_adapter(model, config)
            model.to(device)
        self.model = model
        self.noise = WeightNoise(model, config["noise"]["kind"], config["noise"]["units"],
                                 config["noise"].get("rank", 8))
        self.device = device
        self.details = {"backend": "transformers", "model_class": type(model).__name__,
                        "attention": getattr(model.config, "_attn_implementation", None),
                        "chat_template": self.tokenizer.chat_template,
                        "generation_config": model.generation_config.to_dict()}

    def prepare(self, prompt):
        inputs = encode_prompt(self.tokenizer, prompt, self.config["prompt"]["protocol"])
        return {"prompt": prompt, "input_ids": inputs.tolist()[0],
                "decoded_input": self.tokenizer.decode(inputs[0]),
                "token_sha256": digest(json.dumps(inputs.tolist()).encode())}

    @torch.inference_mode()
    def predict(self, prepared, generation_seed):
        inputs = torch.tensor([prepared["input_ids"]], dtype=torch.long, device=self.device)
        evaluation = self.config["evaluation"]
        if evaluation["method"] == "next_token_exact":
            logits = self.model(inputs).logits[0, -1, :]
            # Preserve the source's dtype-specific softmax followed by argmax.
            token = int(logits.softmax(dim=-1).argmax())
            return {"output": self.tokenizer.decode(token), "output_token_ids": [token]}
        settings = dict(evaluation["generation"])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        settings["pad_token_id"] = self.tokenizer.pad_token_id
        if evaluation["method"] == "legacy_cot":
            from transformers import StoppingCriteria, StoppingCriteriaList
            stop_ids = [self.tokenizer.encode(w, add_special_tokens=False)[0]
                        for w in ["</s>", "Human:", "Assistant:"]]

            class SourceStop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return int(input_ids[0, -1]) in stop_ids

            settings["stopping_criteria"] = StoppingCriteriaList([SourceStop()])
        devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(generation_seed)
            result = self.model.generate(inputs, attention_mask=(inputs != self.tokenizer.pad_token_id).long(), **settings)
        tokens = result[0, inputs.shape[1]:].tolist()
        return {"output": self.tokenizer.decode(tokens, skip_special_tokens=True),
                "full_text": self.tokenizer.decode(result[0], skip_special_tokens=True), "output_token_ids": tokens}


class VLLMModel:
    """Extended provider's real vLLM/PEFT path, with two documented API repairs.

    A CPU checkpoint creates the source-initialized PEFT factors once; only their
    small snapshots survive. vLLM owns the immutable base model thereafter. Zero
    noise uses no adapter request, fixing the upstream missing-adapter error.
    """
    def __init__(self, config, output):
        from peft import get_peft_model_state_dict
        from vllm import LLM
        self.config, self.output = config, Path(output)
        model, self.tokenizer = load_checkpoint(config, "cpu")
        if config["model"].get("adapter_id"):
            raise ValueError("vLLM adapter-on-merged-adapter needs an explicitly exported checkpoint; use legacy WMDP HF preset")
        model = attach_noise_adapter(model, config)
        self.noise = WeightNoise(model, "lora", config["noise"]["units"], config["noise"]["rank"])
        self.layout_record = self.noise.layout()
        self.adapter_config = model.peft_config["default"].to_dict()
        self.adapter_state = get_peft_model_state_dict(model)
        # Keep only tiny adapter parameters for perturbation, not the full base.
        adapter_parameters = [(i, n, p) for i, n, p in self.noise.parameters]
        self.noise.model = None
        del model
        gc.collect()
        self.noise.parameters = adapter_parameters
        spec = config["model"]
        self.engine = LLM(model=spec["id"], revision=revision(spec["id"]),
                          tokenizer=spec.get("tokenizer_id") or spec["id"],
                          tokenizer_revision=revision(spec.get("tokenizer_id") or spec["id"]),
                          dtype=spec["dtype"], enable_lora=True, max_lora_rank=config["noise"]["rank"],
                          seed=42, tensor_parallel_size=spec.get("tensor_parallel_size", 1),
                          trust_remote_code=spec.get("trust_remote_code", False),
                          download_dir=str(ROOT / "models/hf"),
                          max_model_len=spec["max_model_len"])
        self.details = {"backend": "vllm", "chat_template": self.tokenizer.chat_template,
                        "tensor_parallel_size": spec.get("tensor_parallel_size", 1)}
        self.request = None
        self.request_number = 0

    def prepare(self, prompt):
        text = format_text(self.tokenizer, prompt)
        ids = self.tokenizer.encode(text)
        return {"prompt": prompt, "text": text, "input_ids": ids, "decoded_input": self.tokenizer.decode(ids),
                "token_sha256": digest(json.dumps([ids]).encode())}

    def set_adapter(self, magnitude):
        from safetensors.torch import save_file
        from vllm.lora.request import LoRARequest
        if magnitude == 0:
            self.request = None
            return
        self.request_number += 1
        path = self.output / "active_adapter"
        path.mkdir(exist_ok=True)
        # PEFT state keys omit '.default'. Copy the current perturbed factors.
        state = {n.replace(".default", ""): p.detach().cpu().contiguous()
                 for _, n, p in self.noise.parameters}
        save_file(state, str(path / "adapter_model.safetensors"))
        clean = json.loads(json.dumps(self.adapter_config, default=lambda v: sorted(v) if isinstance(v, set) else str(v)))
        (path / "adapter_config.json").write_text(json.dumps(clean))
        self.request = LoRARequest(f"noise_{self.request_number}", self.request_number, str(path.resolve()))

    def predict(self, prepared, generation_seed):
        from vllm import SamplingParams
        settings = self.config["evaluation"]["vllm_sampling"]
        result = self.engine.generate(prepared["text"], SamplingParams(seed=generation_seed, **settings),
                                      lora_request=self.request, use_tqdm=False)[0].outputs[0]
        return {"output": result.text, "output_token_ids": list(result.token_ids)}
