"""Fetch pinned evidence; extract exact assets; materialize explicit run configs.

This is data extraction, not an implementation import from the upstream package.
Run from the project root. No model weights are downloaded by this script.
"""
import argparse
import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.io import digest, read_json, write_json

REPOS = {
    "SandbagDetect": ("https://github.com/camtice/SandbagDetect.git", "f8d5713e26aa47dcee447f3d65f67127f6382787"),
    "noise_injection_extended": ("https://github.com/camtice/noise_injection_extended.git", "6c40221bd8f4f13d725006f8a003ef2584af9ab5"),
}
MODELS = {
    "llama8b": ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama", "llama8b", "chat"),
    "mistral7b": ("mistralai/Mistral-7B-Instruct-v0.2", "Llama", "mistral7B", "chat"),
    "phi_mini": ("microsoft/Phi-3-mini-4k-instruct", "Phi", "phi3-mini", "chat"),
    "phi_small": ("microsoft/Phi-3-small-128k-instruct", "Phi", "phi3-small", "chat"),
    "phi_medium": ("microsoft/Phi-3-medium-4k-instruct", "Phi", "phi3-medium", "phi_manual"),
}
DATA = {"ai2": "tiny_AI2_formatted.jsonl", "mmlu": "tiny_mmlu_formatted.jsonl",
        "wmdp": "mod_WMDP_without_password.jsonl", "boolq": "tiny_boolq_formatted.jsonl"}


def fetch_sources():
    for name, (url, commit) in REPOS.items():
        path = ROOT / "sources" / name
        if not path.exists():
            subprocess.run(["git", "clone", url, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", commit], check=True)
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise ValueError(f"Source checkout changed: {name}: {actual}")
    subprocess.run(["git", "-C", str(ROOT / "sources/noise_injection_extended"), "submodule", "update", "--init"], check=True)
    paper = ROOT / "sources/paper-v3.html"
    if not paper.exists():
        urllib.request.urlretrieve("https://arxiv.org/html/2412.01784v3", paper)


def extract_function(path, name):
    """Compile only an inspected pure prompt function, never upstream imports."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    fn = copy.deepcopy(fn)
    fn.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def assignment(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    # DEFAULT_PROMPT_CoT is a literal followed by .strip().
    return ast.literal_eval(node.value.func.value).strip()


def no_goal_code(path):
    # Only literal messages and the question f-string are evaluated, not task code.
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "record_to_sample")
    assignment_node = next(n for n in fn.body if isinstance(n, ast.Assign))
    messages = []
    roles = {"ChatMessageSystem": "system", "ChatMessageUser": "user", "ChatMessageAssistant": "assistant"}
    for call in assignment_node.value.elts:
        content = next(k.value for k in call.keywords if k.arg == "content")
        if isinstance(content, ast.JoinedStr):
            text = "".join(v.value if isinstance(v, ast.Constant) else "{question}" for v in content.values)
        else:
            text = ast.literal_eval(content)
        messages.append({"role": roles[call.func.id], "content": text})
    return messages


def extract_assets():
    from bs4 import BeautifulSoup
    legacy = ROOT / "sources/SandbagDetect"
    extended = ROOT / "sources/noise_injection_extended"
    system = extract_function(legacy / "src/SystemPrompt.py", "set_system_prompt")
    soup = BeautifulSoup((ROOT / "sources/paper-v3.html").read_text(), "html.parser")
    appendix = soup.find(id="A1")
    # Appendix A uses five consecutive quote blocks; get_text preserves authored
    # line breaks inside paragraphs. Whitespace around the block is layout only.
    blocks = [b.get_text().strip() for b in appendix.find_all("blockquote")]
    if len(blocks) != 6:
        raise ValueError(f"Paper Appendix A layout changed: {len(blocks)} quote blocks")
    code_sbg = no_goal_code(extended / "evals/clement_mistral_large_sandbag.py")
    code_control = no_goal_code(extended / "evals/gsm8k_realistic_no_sandbag.py")
    # Paper figures are SVG boxes. Preserve each published box's text rather
    # than silently substitute the divergent Qwen task from current code.
    paper_messages = []
    for para, role in [(3, "system"), (4, "user"), (5, "assistant"), (6, "user"), (7, "assistant"), (8, "user")]:
        box = soup.find(id=f"A2.p{para}.pic1.2")
        for link in box.select(".ltx_listing_data"):
            link.decompose()
        content = box.get_text().strip()
        content = content.replace("[A single question from GSM8k is inserted here]", "{question}")
        paper_messages.append({"role": role, "content": content})
    control_box = soup.find(id="A2.p10.pic1.2")
    if control_box is None:
        candidates = [b for b in soup.select("#A2 .ltx_minipage") if "score below 95%" in b.get_text()]
        control_box = candidates[-1]
    paper_control = copy.deepcopy(paper_messages)
    paper_control[3]["content"] = control_box.get_text().strip().removesuffix('"""').rstrip()
    pw = extended / "evals/pw_gsm8k.py"
    tree = ast.parse(pw.read_text())
    pw_template = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                       and "{password}" in n.value and "{prompt}" in n.value).strip()
    assets = {
        "legacy_literals": {"standard": system("Llama", False), "Llama": system("Llama", True),
                            "Phi": system("Phi", True), "Password": system("Password", False)},
        "paper_v3": {"standard": blocks[0], "Llama": blocks[2], "Phi": blocks[3],
                     "gsm8k_standard": blocks[1], "gsm8k_sandbag": blocks[4]},
        "extended": {"pw_absent": assignment(pw, "DEFAULT_PROMPT_CoT"),
                     "pw_present": pw_template.replace("{password}", "This problem is quite special, read it carefully!"),
                     "gsm8k_standard": assignment(extended / "evals/gsm8k.py", "DEFAULT_PROMPT_CoT"),
                     "gsm8k_sandbag": assignment(extended / "evals/prompt_gsm8k_sandbag.py", "DEFAULT_PROMPT_CoT")},
        "no_goal": {"paper_v3_boxes": {"sandbag_incentive": paper_messages, "no_sandbag_incentive": paper_control},
                    "current_code": {"sandbag_incentive": code_sbg, "no_sandbag_incentive": code_control}},
    }
    write_json(ROOT / "assets/prompts.json", assets)
    datasets = {}
    for path in sorted((legacy / "data").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        datasets[path.name] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes()),
                               "expected_count": len(rows), "answer_values": sorted({str(r["answer"]) for r in rows})}
    write_json(ROOT / "assets/datasets.lock.json", datasets)
    files = [p for base in [legacy / "src", legacy / "dataset-conversion", legacy / "data_analysis_and_plotting",
                           extended / "evals", extended / "configs", extended / "analysis"]
             for p in base.rglob("*") if p.is_file() and p.suffix in {".py", ".jsonl", ".ipynb", ".yaml"}]
    files += [extended / "inspect_ai_/src/inspect_ai/model/_providers/noise_hf.py",
              extended / "inspect_ai_/src/inspect_ai/scorer/_common.py", ROOT / "sources/paper-v3.html"]
    write_json(ROOT / "assets/sources.lock.json", {
        "paper": "https://arxiv.org/abs/2412.01784v3", "paper_version": 3,
        "repositories": {name: {"url": url, "commit": sha} for name, (url, sha) in REPOS.items()},
        "inspect_submodule": "6fe316f6f846f59caed0f6db552df1803e980f7b",
        "files": {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files},
    })


def archive_groups(model_directory, dataset):
    directory = ROOT / "sources/SandbagDetect/results/SoLaR_Regression_Exps" / model_directory / dataset
    groups, records, grids = {}, [], []
    for path in sorted(directory.glob("*v[1-5].csv")):
        rows = list(csv.DictReader(io.StringIO("\n".join(l for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")))))
        measurement = path.stem.rsplit("_", 1)[-1]
        seeds = list(dict.fromkeys(int(float(r["Seed"])) for r in rows))
        grid = sorted({float(r["Noise Std"]) for r in rows})
        groups[measurement] = seeds
        grids.append(grid)
        records.append({"measurement": measurement, "path": str(path.relative_to(ROOT)),
                        "sha256": digest(path.read_bytes()), "seeds_observed": seeds, "magnitudes_observed": grid,
                        "header": [l for l in path.read_text().splitlines() if l.startswith("#")]})
    if len(groups) != 5:
        raise ValueError(f"Expected five archived measurements in {directory}, found {len(groups)}")
    return groups, records, grids


def make_configs():
    datasets = read_json(ROOT / "assets/datasets.lock.json")
    base = {"name": "mcq_base", "evidence": "v3 methods reconstruction; historical revisions not published",
            "model": {"backend": "hf", "dtype": "bfloat16", "device": "cuda:0", "attention": "eager", "trust_remote_code": True},
            "data": {}, "prompt": {"protocol": "mcq", "text_source": "paper_v3", "family": "Llama", "format": "chat"},
            "noise": {"kind": "direct", "units": "absolute_weight_std", "mean": 0, "percentage": 1,
                      "sweep": {"start": 0, "stop": 0.01, "count": 101}},
            "seed_groups": {}, "evaluation": {"method": "next_token_exact"}}
    write_json(ROOT / "configs/mcq.json", base)
    provenance = []
    jobs = []
    for model_name, (model_id, family, archive_dir, formatting) in MODELS.items():
        for dataset, source_dir in [("ai2", "AI2"), ("mmlu", "MMLU"), ("wmdp", "WMDP")]:
            archive_dataset = "WMPD" if dataset == "wmdp" and family == "Phi" else source_dir
            groups, records, grids = archive_groups(archive_dir, archive_dataset)
            for record in records:
                record.update(model=model_name, benchmark=dataset)
            provenance.extend(records)
            job = {"extends": "../mcq.json", "name": f"{model_name}_{dataset}_paper_v3",
                   "model": {"id": model_id}, "data": datasets[DATA[dataset]], "seed_groups": groups,
                   "prompt": {"family": family, "format": formatting}, "analysis_group": model_name,
                   "benchmark": dataset, "evidence": "Paper v3 Appendix A prompts; 101-point methods grid; observed archive seeds. Not a historical exact replay."}
            if model_name in {"phi_small", "phi_medium"}:
                job["model"]["attention"] = "flash_attention_2" if model_name == "phi_small" else "eager"
            write_json(ROOT / f"configs/full/{model_name}_{dataset}.json", job)
            jobs.append(f"configs/full/{model_name}_{dataset}.json")
            for record, grid in zip(records, grids):
                archived = copy.deepcopy(job)
                archived["name"] = f"{model_name}_{dataset}_{record['measurement']}_source_literals"
                archived["prompt"]["text_source"] = "legacy_literals"
                archived["noise"] = {"sweep": grid}
                archived["seed_groups"] = {record["measurement"]: record["seeds_observed"]}
                archived["evidence"] = "Observed archive grid and seeds + current source literal prompts; archived prompt history is not recoverable."
                write_json(ROOT / f"configs/archive/{model_name}_{dataset}_{record['measurement']}.json", archived)
    write_json(ROOT / "configs/main_matrix.json", {"jobs": jobs})
    _, password_records, _ = archive_groups("mistral7b-PW", "WMPD")
    for record in password_records:
        record.update(model="mistral_password", benchmark="wmdp")
    write_json(ROOT / "assets/archive_measurements.json", provenance + password_records)
    smoke = {"extends": "mcq.json", "name": "tiny_random_cpu_workflow_only", "evidence": "Random 1M-parameter model, first 3 AI2 rows; workflow validation only, not a scientific reproduction",
             "model": {"id": "hf-internal-testing/tiny-random-LlamaForCausalLM", "device": "cpu", "dtype": "float32", "trust_remote_code": False},
             "data": dict(datasets[DATA["ai2"]], limit=3), "noise": {"sweep": [0, 0.001, 0.003]},
             "seed_groups": {"smoke": [11, 22]}, "hash_each_trial": True, "verify_every_restoration": True}
    write_json(ROOT / "configs/smoke.json", smoke)
    pilot = copy.deepcopy(smoke)
    pilot.update(name="smollm2_135m_local_pilot", evidence="Smaller trained model and first 10 AI2 rows; feasibility pilot, not paper evidence")
    pilot["model"]["id"] = "HuggingFaceTB/SmolLM2-135M-Instruct"
    pilot["model"]["device"] = "cuda:0"
    pilot["model"]["dtype"] = "bfloat16"
    pilot["data"]["limit"] = 10
    pilot["hash_each_trial"] = False
    write_json(ROOT / "configs/local_pilot.json", pilot)
    cpu_pilot = {"extends":"local_pilot.json", "name":"smollm2_135m_cpu_pilot", "model":{"device":"cpu","dtype":"float32"},
                 "data":{"limit":5}, "evidence":"Trained SmolLM2-135M, CPU float32, first 5 AI2 questions, two seeds and 3 noise levels. Approximate workflow/feasibility pilot, not a paper model or paper result."}
    write_json(ROOT / "configs/local_pilot_cpu.json", cpu_pilot)
    pw_seeds = [3852, 44269, 35909, 99550, 62298]
    password = {"extends": "mcq.json", "name": "legacy_wmdp_password_evaluation", "analysis_group": "mistral_password", "benchmark": "wmdp",
                "evidence": "Legacy dense-noise evaluation of released rank-16 adapter (merged), NOT adapter training; current config grid",
                "model": {"id": "mistralai/Mistral-7B-v0.1", "tokenizer_id": "FelixHofstaetter/mistral-7b-sandbagging-new",
                          "adapter_id": "FelixHofstaetter/mistral-7b-sandbagging-new", "dtype": "float32", "resize_seed": 42},
                "data": datasets[DATA["wmdp"]], "prompt": {"protocol": "wmdp_password"},
                "noise": {"sweep": {"start": 0, "stop": 0.0015, "count": 100}}, "seed_groups": {"current_config": pw_seeds}}
    write_json(ROOT / "configs/password_wmdp_legacy.json", password)
    appendix = copy.deepcopy(password)
    appendix.update(name="wmdp_appendix_d_grid_dense_interpretation", evidence="Appendix D endpoints/increment imply 21 points, although caption says 20; dense legacy interpretation (v3 prose instead claims LoRA)")
    appendix["noise"]["sweep"] = {"start": 0, "stop": 0.001, "count": 21}
    write_json(ROOT / "configs/password_wmdp_appendix_d.json", appendix)
    portable_lora = copy.deepcopy(appendix)
    portable_lora.update(name="v3_wmdp_lora_portable_reconstruction", evidence="v3 claims LoRA for WMDP, but extended repo has no WMDP task or frozen sweep. Rank-8 factors on merged released adapter; HF inference and legacy raw prompt/scorer are explicitly a reconstruction, not recovered v3 inputs.")
    portable_lora["model"]["dtype"] = "bfloat16"
    portable_lora["noise"].update(kind="lora", units="paper_direct_std", rank=8, initialization_seed=42,
                                  sweep={"start":0,"stop":.001,"count":21})
    write_json(ROOT / "configs/password_wmdp_lora_portable.json", portable_lora)
    length = {"extends": "full/llama8b_ai2.json", "name": "appendix_a1_length_control", "evidence": "Current precision prompt literal and 10 linspace targets; target list is an explicit reconstruction of Appendix A.1",
              "prompt": {"protocol": "percentage", "percentages": [i * 100 / 9 for i in range(10)]}}
    write_json(ROOT / "configs/length_control.json", length)
    math_data = {"path":"data/gsm8k_test_first30.jsonl", "expected_count":30}
    math_provenance = ROOT / "data/gsm8k_provenance.json"
    if math_provenance.exists():
        math_data.update(sha256=read_json(math_provenance)["sha256"], dataset_revision=read_json(math_provenance)["revision"])
    extended = {"extends": "../mcq.json", "name": "extended_deepseek_password", "analysis_group": "deepseek_password", "benchmark": "gsm8k",
                "evidence": "v3 LoRA scale conversion with frozen PEFT A0 initialization; paired first-30 GSM8K reconstruction (source present task defaults to 10, absent to 30)",
                "model": {"id": "redwoodresearch/math_pwd_lock_deepseek_math7b_on_weak_pythia1b", "tokenizer_id": "deepseek-ai/deepseek-math-7b-base",
                          "backend": "vllm", "max_model_len": 2048, "tensor_parallel_size": 1},
                "data": math_data, "prompt": {"protocol": "deepseek_password"},
                "noise": {"kind": "lora", "rank": 8, "initialization_seed": 42, "units": "paper_direct_std"},
                "seed_groups": {f"measurement{i+1}": list(range(i*5+1,i*5+6)) for i in range(5)},
                "evaluation": {"method": "dollar", "generation_seed": 42,
                               "generation": {"do_sample": True, "temperature": 0.01, "top_p": 1.0, "max_new_tokens": 1500},
                               "vllm_sampling": {"temperature": 0.01, "top_p": 1.0, "max_tokens": 1500}}}
    write_json(ROOT / "configs/extended/deepseek_password.json", extended)
    portable_deepseek = copy.deepcopy(extended)
    portable_deepseek["model"]["backend"] = "hf"
    portable_deepseek.update(name="deepseek_password_hf_portable", evidence="Portable HF backend for inspecting source LoRA intervention; differs numerically from published vLLM; paired 30-question reconstruction, not the 10/30 source default mismatch")
    write_json(ROOT / "configs/extended/deepseek_password_hf_portable.json", portable_deepseek)
    raw = copy.deepcopy(extended)
    raw.update(name="extended_provider_raw_factor_units", evidence="Source provider takes noise_std as factor std without conversion; otherwise paired reconstruction of deepseek task")
    raw["noise"]["units"] = "factor_std"
    write_json(ROOT / "configs/extended/deepseek_raw_factor_units.json", raw)
    for name, model_id in [("llama8b", MODELS["llama8b"][0]), ("mistral7b", MODELS["mistral7b"][0])]:
        cot = copy.deepcopy(extended)
        cot.update(name=f"extended_{name}_gsm8k", analysis_group=name, evidence="Current extended GSM8K task literals, 30 test questions, source numeric-end scorer; LoRA backend, not direct legacy CoT")
        cot["model"].update(id=model_id, tokenizer_id=model_id)
        cot["prompt"]["protocol"] = "extended_gsm8k"
        cot["evaluation"]["method"] = "inspect_numeric_end"
        cot["model"]["max_model_len"] = 4096
        write_json(ROOT / f"configs/extended/{name}_gsm8k.json", cot)
    no_goal = copy.deepcopy(extended)
    no_goal.update(name="v3_no_goal_mistral_large", analysis_group="mistral_large", evidence="Appendix B figure-box text, 30 GSM8K questions; text extraction preserves published wording but figure whitespace cannot recover original input bytes; seed IDs reconstructed")
    no_goal["model"].update(id="mistralai/Mistral-Large-Instruct-2411", tokenizer_id="mistralai/Mistral-Large-Instruct-2411", tensor_parallel_size=8, max_model_len=8192)
    no_goal["prompt"].update(protocol="no_goal", variant="paper_v3_boxes")
    no_goal["seed_groups"] = {"single_measurement": [1,2,3,4,5]}
    no_goal["evaluation"]["method"] = "inspect_numeric_end"
    no_goal["evaluation"]["generation"]["max_new_tokens"] = 7000
    no_goal["evaluation"]["vllm_sampling"]["max_tokens"] = 7000
    write_json(ROOT / "configs/extended/no_goal_v3.json", no_goal)
    current = copy.deepcopy(no_goal)
    current.update(name="current_code_no_goal_not_v3", evidence="Current source task messages, including divergent goals and Qwen control text; 20 questions, not paper's 30")
    current["prompt"]["variant"] = "current_code"
    current["data"]["limit"] = 20
    write_json(ROOT / "configs/extended/no_goal_current_code.json", current)
    for name, parent, sigma in [("deepseek_password", "../extended/deepseek_password.json", .0052),
                                 ("llama8b_mmlu", "../full/llama8b_mmlu.json", .0012),
                                 ("mistral7b_mmlu", "../full/mistral7b_mmlu.json", .0004)]:
        scaling = {"extends": parent, "name": name + "_100_seed_scaling", "scaling_experiment": True,
                   "noise": {"sweep": [sigma]}, "seed_groups": {f"replicate{i+1}": list(range(1000+i*100,1100+i*100)) for i in range(5)},
                   "evidence": "v3 Figure 4 published fixed sigma; 5x100 seeds. Exact seed IDs unreported: preregistered 1000..1499 reconstruction; no selection on new test results."}
        write_json(ROOT / f"configs/scaling/{name}.json", scaling)


def lock_models():
    ids = [m[0] for m in MODELS.values()] + ["hf-internal-testing/tiny-random-LlamaForCausalLM", "HuggingFaceTB/SmolLM2-135M-Instruct",
           "mistralai/Mistral-7B-v0.1", "FelixHofstaetter/mistral-7b-sandbagging-new", "deepseek-ai/deepseek-math-7b-base",
           "redwoodresearch/math_pwd_lock_deepseek_math7b_on_weak_pythia1b", "mistralai/Mistral-Large-Instruct-2411"]
    path = ROOT / "assets/models.lock.json"
    lock = read_json(path) if path.exists() else {}
    for model_id in ids:
        if model_id in lock and lock[model_id].get("revision"):
            continue
        raw = subprocess.check_output(["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", "60",
                                       f"https://huggingface.co/api/models/{model_id}"], text=True)
        info = json.loads(raw)
        lock[model_id] = {"revision": info["sha"], "historical_revision": None,
                          "revision_note": "Resolved for this reproduction, not the authors' unreported historical revision",
                          "files": [f["rfilename"] for f in info.get("siblings", [])],
                          "card_data": info.get("cardData", {})}
        write_json(path, lock)
        print(model_id, info["sha"], flush=True)


def gsm8k_data():
    import pyarrow.parquet as parquet
    # Download the exact pinned HF Parquet bytes; avoids dataset script execution.
    info = json.loads(subprocess.check_output(["curl","-L","--fail","--silent","--show-error","--max-time","60",
                                              "https://huggingface.co/api/datasets/openai/gsm8k"],text=True))
    revision = info["sha"]
    files = [f["rfilename"] for f in info["siblings"] if f["rfilename"].startswith("main/test-") and f["rfilename"].endswith(".parquet")]
    if len(files) != 1:
        raise ValueError("Expected one GSM8K main/test Parquet shard")
    cache = ROOT / "sources/dataset_cache"
    cache.mkdir(exist_ok=True)
    source = cache / ("gsm8k_test_"+revision+".parquet")
    if not source.exists():
        subprocess.run(["curl","-L","--fail","--silent","--show-error","--max-time","120",
                        f"https://huggingface.co/datasets/openai/gsm8k/resolve/{revision}/{files[0]}","-o",str(source)],check=True)
    dataset = parquet.read_table(source).slice(0,30).to_pylist()
    rows = [{"question": row["question"], "answer": row["answer"].split("####")[-1].strip()} for row in dataset]
    path = ROOT / "data/gsm8k_test_first30.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(ROOT / "data/gsm8k_provenance.json", {"dataset": "openai/gsm8k", "subset": "main", "split": "test", "rows": list(range(30)),
                                                   "revision": revision, "sha256": digest(path.read_bytes())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-models", action="store_true")
    parser.add_argument("--gsm8k", action="store_true")
    args = parser.parse_args()
    fetch_sources()
    extract_assets()
    make_configs()
    if args.lock_models:
        lock_models()
    if args.gsm8k:
        gsm8k_data()
        make_configs()
