import copy
import torch
import pytest
from repro.config import load_config
from repro.io import ROOT, read_json
from repro.noise import WeightNoise
from repro import runner
from repro.analysis import run_rows


class RecordedModel:
    """Unit fixture, not a research backend or a source of saved research results."""
    instances = []

    def __init__(self, config):
        self.model = torch.nn.Linear(2,2)
        self.noise = WeightNoise(self.model)
        self.details = {"backend":"unit_fixture"}
        self.calls = []
        self.instances.append(self)

    def prepare(self,prompt):
        return {"prompt":prompt,"input_ids":[1],"token_sha256":"unit"}

    def predict(self,prepared,generation_seed):
        self.calls.append((self.noise.fingerprint(),generation_seed))
        return {"output":"A", "output_token_ids":[1]}


def test_pairing_restoration_and_complete_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(runner,"HFModel",RecordedModel)
    config=load_config(ROOT/"configs/smoke.json")
    runner.run(config,tmp_path)
    backend=RecordedModel.instances[-1]
    # Six predictions per noisy cell: the two conditions share one fingerprint.
    for start in range(0,len(backend.calls),6):
        assert len({h for h,s in backend.calls[start:start+6]})==1
    backend.noise.assert_restored()
    rows,predictions=run_rows(tmp_path)
    assert len(rows)==12
    assert len(predictions)==36
    runner.run(config,tmp_path)
    assert not RecordedModel.instances[-1].calls
    changed=copy.deepcopy(config)
    changed["name"]="different"
    with pytest.raises(ValueError,match="Resume identity"):
        runner.run(changed,tmp_path)


def test_renamed_repeated_seeds_are_rejected_as_nonindependent(tmp_path, monkeypatch):
    from repro.analysis import analyze
    monkeypatch.setattr(runner,"HFModel",RecordedModel)
    first=load_config(ROOT/"configs/smoke.json")
    second=copy.deepcopy(first)
    second["seed_groups"]={"renamed_smoke":[11,22]}
    runner.run(first,tmp_path/"first")
    runner.run(second,tmp_path/"second")
    with pytest.raises(ValueError,match="not independent"):
        analyze([tmp_path/"first",tmp_path/"second"],tmp_path/"analysis")


def test_dropped_question_is_rejected_even_with_updated_checksum(tmp_path,monkeypatch):
    from repro.io import write_json,json_digest
    monkeypatch.setattr(runner,"HFModel",RecordedModel)
    config=load_config(ROOT/"configs/smoke.json")
    runner.run(config,tmp_path)
    path=next((tmp_path/"trials").glob("*.json"))
    checkpoint=read_json(path)
    checkpoint["trial"]["predictions"].pop()
    checkpoint["checksum"]=json_digest(checkpoint["trial"])
    write_json(path,checkpoint)
    with pytest.raises(ValueError,match="Missing or repeated"):
        run_rows(tmp_path)


def test_real_tiny_transformer_zero_noise_logits_and_restore():
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.manual_seed(3)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                                      num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).eval()
    noise=WeightNoise(model)
    inputs=torch.tensor([[1,2,3,4]])
    with torch.inference_mode():
        baseline=model(inputs).logits.clone()
        noise.apply(.1,8)
        assert not torch.equal(baseline,model(inputs).logits)
        noise.apply(0,8)
        assert torch.equal(baseline,model(inputs).logits)
    noise.assert_restored()
