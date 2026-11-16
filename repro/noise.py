"""The two published interventions, with exact copy-based restoration.

Direct: W_sigma = W_0 + Normal(0, sigma), including embeddings/norms/head.
LoRA: A_sigma = A_0 + N_A, B_sigma = B_0 + N_B, alpha/r = 1.
The effective update is B_sigma @ A_sigma; A_0 is NOT zero in PEFT.
"""
from contextlib import contextmanager
import hashlib
import math
import torch


def tensor_hash(tensor):
    # View bytes, rather than convert dtype: bfloat16 is not a NumPy dtype.
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def factor_std(magnitude, units, rank=8):
    if units == "factor_std":
        return magnitude
    if units == "paper_direct_std":
        return rank ** 0.25 * math.sqrt(magnitude)
    raise ValueError(f"Unknown LoRA units: {units}")


class WeightNoise:
    """Own the clean snapshot; never restore by subtracting rounded noise.

    Direct generators live on the parameter device and use seed + named index,
    exactly as add_noise_all. LoRA uses one CPU stream in named-parameter order,
    as the extended provider's temporary CPU model does. Neither stream mutates
    PyTorch's global generator or depends on evaluation/condition ordering.
    """
    def __init__(self, model, kind="direct", units="absolute_weight_std", rank=8):
        self.model, self.kind, self.units, self.rank = model, kind, units, rank
        self.parameters = []
        for index, (name, parameter) in enumerate(model.named_parameters()):
            selected = kind == "direct" or "lora_A" in name or "lora_B" in name
            if selected:
                if not parameter.is_floating_point() or parameter.device.type == "meta":
                    raise ValueError(f"Cannot perturb non-floating or offloaded tensor {name}")
                self.parameters.append((index, name, parameter))
        if not self.parameters:
            raise ValueError("No parameters selected")
        self.originals = {name: p.detach().cpu().clone() for _, name, p in self.parameters}

    @torch.no_grad()
    def restore(self):
        for _, name, parameter in self.parameters:
            parameter.copy_(self.originals[name])

    def assert_restored(self):
        for _, name, p in self.parameters:
            if not torch.equal(p.detach().cpu().reshape(-1).view(torch.uint8), self.originals[name].reshape(-1).view(torch.uint8)):
                raise AssertionError(f"Restoration differs bitwise: {name}")

    def layout(self):
        return [{"index": i, "name": n, "shape": list(p.shape), "dtype": str(p.dtype),
                 "device": str(p.device), "elements": p.numel(), "clean_sha256": tensor_hash(self.originals[n])}
                for i, n, p in self.parameters]

    @torch.no_grad()
    def apply(self, magnitude, seed):
        if magnitude < 0 or not math.isfinite(magnitude):
            raise ValueError("Magnitude must be finite and nonnegative")
        self.restore()
        if magnitude == 0:
            return
        std = magnitude if self.kind == "direct" else factor_std(magnitude, self.units, self.rank)
        lora_generator = torch.Generator(device="cpu").manual_seed(seed)
        for index, _, parameter in self.parameters:
            device = parameter.device if self.kind == "direct" else torch.device("cpu")
            generator = (torch.Generator(device=device).manual_seed(seed + index)
                         if self.kind == "direct" else lora_generator)
            noise = torch.normal(0.0, std, size=parameter.shape, dtype=parameter.dtype,
                                 device=device, generator=generator)
            parameter.add_(noise.to(parameter.device))

    @contextmanager
    def trial(self, magnitude, seed):
        try:
            self.apply(magnitude, seed)
            yield self
        finally:
            self.restore()

    def fingerprint(self):
        return hashlib.sha256("".join(tensor_hash(p) for _, _, p in self.parameters).encode()).hexdigest()

