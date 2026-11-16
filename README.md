# Noise Injection and Sandbagging: Research Reproduction

A readable research implementation inspired by [Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models, arXiv v3](https://arxiv.org/abs/2412.01784v3). It investigates whether adding Gaussian noise to model weights can reveal capabilities that are hidden by deliberate underperformance, known as *sandbagging*.

**Status: work in progress, not a completed reproduction of the paper.** Small CPU experiments and automated checks have run; the full paper-scale GPU experiments have not. Some configurations are explicitly reconstructions, and the audit below identifies problems to resolve before spending money on large runs. Published author results and new local results are labeled separately.

## The research question

The paper compares an ordinary model with the same model under a sandbagging condition. It evaluates both without noise, then repeatedly adds Gaussian noise and measures how accuracy changes. In some settings, a sandbagging model's accuracy improves under noise before eventually degrading.

This project supports several distinct conditions:

| Condition | Meaning |
| --- | --- |
| Ordinary | The model is instructed to answer accurately. |
| Prompted sandbagging | An instruction asks the model to underperform. |
| Password present / absent | An existing password-locked model is evaluated with or without the capability-eliciting phrase. This project evaluates the released model; it does not reproduce its training. |
| No-goal incentive | A reconstructed setting changes environmental incentives without directly requesting underperformance. The paper and current released source differ; the configurations keep these interpretations separate. |

For direct perturbations, the implementation adds absolute Gaussian noise to model parameters, starting from a saved clean copy for each trial. For LoRA perturbations, it perturbs adapter factors rather than substituting activation noise or sampling-temperature changes. Conditions in a paired trial are evaluated using the same perturbed weights and questions. See [the experimental specification](docs/EXPERIMENTS.md) for equations, source differences, and the limitations of comparing direct and LoRA noise.

A measurement uses the improvement ratio `phi = maximum accuracy across the selected noise sweep / zero-noise accuracy`. A large ratio in a tiny pilot alone is not evidence of sandbagging: scores can change dramatically when a dataset contains only a few questions.

## Project layout

```text
.
├── README.md                   Project overview and getting started
├── ATTRIBUTION.md              Upstream attribution and licensing boundaries
├── configs/                    Experiment definitions and inheritance
│   ├── full/                   Prompted multiple-choice model/benchmark matrix
│   ├── archive/                Configurations using observed historical grids
│   ├── extended/               Math, password, and no-goal reconstructions
│   └── scaling/                Fixed-noise seed-scaling presets (see known issues)
├── repro/                      Experiment implementation and command-line interface
│   ├── runner.py               Paired trial loop and per-question records
│   ├── models.py               Hugging Face and vLLM model paths
│   ├── noise.py                Gaussian interventions and weight restoration
│   ├── prompts.py              Condition-specific prompt construction
│   ├── scoring.py              Source scoring and corrected diagnostics
│   ├── analysis.py             Summaries, statistics, detection, and plots
│   ├── config.py               Configuration loading and validation
│   └── checks.py               Model and intervention preflight checks
├── scripts/                    Source preparation, validation, paper comparison
├── tests/                      Focused automated checks
├── assets/                     Extracted prompt text and source/model/data locks
├── data/                       Prepared small data files and provenance
├── docs/                       Detailed experimental specification and results
├── results/                    Saved validation, pilots, and author-data analysis
├── sources/                    Upstream metadata; downloaded source checkouts are ignored
├── requirements*.txt           Separate dependency profiles
└── runs/                       New run outputs, created locally and ignored by Git
```

The downloaded model cache (`models/`), virtual environments, upstream repository checkouts, and new `runs/` outputs are not included in Git. On a fresh machine, use the preparation step below to fetch the pinned upstream source checkouts.

## Current results and fidelity

The [results record](docs/RESULTS.md) identifies actual runs, and the [experimental specification](docs/EXPERIMENTS.md) maps paper sections to source files, models, prompts, data, noise, and configurations. Important distinctions:

- **Local validation:** 25 focused tests passed in the previously recorded environment. Small random-model and trained 135M-model CPU pilots completed. This does not validate GPU, vLLM, or Flash Attention paths.
- **Author archive:** [results/author_archive/](results/author_archive/) analyzes the authors' saved CSVs. It does not represent new model inference.
- **Full reproduction:** no full paper-model sweeps, password-model experiments, 100-seed scaling runs, or Mistral Large experiments have been completed here.
- **Reconstruction choices:** original checkpoint revisions and some historical settings are unavailable; the locked model revisions are reproducible choices made for this project, not proven historical revisions.

### Known issues before expensive runs

1. **Scaling presets fail analysis:** configuration inheritance keeps extra five-seed measurement groups along with the intended 100-seed groups. The scaling analysis rejects those inherited groups. Do not launch the full scaling jobs until fixed.
2. **Prompted GSM8K differs from the paper:** the available extended presets use LoRA noise and current released-source prompts. Paper v3 specifies direct weight noise for prompted sandbagging and provides different prompt text. These presets are not a faithful replay of those paper results.
3. **WMDP password v3 reconstruction:** the portable LoRA preset follows the legacy raw-question prompt and omits the helpful-assistant system prompt stated in Appendix D. The legacy behavior and paper description are distinct interpretations.
4. **Mixed-run analysis:** combining some results with different prediction fields can fail depending on input order. The dollar-answer scorer also raises an error for comma-formatted targets instead of returning the upstream scorer's incorrect result; the prepared first 30 GSM8K answers do not contain such targets.
5. **Other boundaries:** length-control targets follow the released code's decimal values rather than the paper's stated integers; two AI2 answer keys are numeric despite lettered choices, with a separately labeled corrected diagnostic; and the 70B quantized intervention remains unresolved.

Passing tests does not establish that these issues are resolved. See [the detailed specification](docs/EXPERIMENTS.md) and [results record](docs/RESULTS.md) for additional caveats. The scaling workflow documented there should not be used until the inherited seed-group issue is fixed.

## Quick start: a small CPU smoke test

Run from the project root on Linux/macOS. Python 3.10–3.12 is the stated support range; the recorded local environment used Python 3.10.12. A fresh setup needs internet access for dependencies, pinned sources, and model downloads.

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU PyTorch and the general analysis environment
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-analysis.txt

# Fetch pinned upstream sources and regenerate extracted assets/configurations.
# This can overwrite generated assets/configs: back up any custom edits first.
python scripts/prepare.py

python scripts/validate.py
python -m repro plan configs/smoke.json
python -m repro check configs/smoke.json --output runs/smoke_preflight.json
python -m repro run configs/smoke.json --output runs/smoke
python -m repro analyze runs/smoke --output runs/smoke_analysis
```

The smoke checkpoint is a tiny **random-weight** model, not a paper model. It checks the plumbing, not the research hypothesis. The committed model lock pins reproduction-time revisions. `python scripts/prepare.py --lock-models` resolves and rewrites that lock; use it only if intentionally refreshing revisions, as that changes the experiment identity. Accept any applicable model licenses and authenticate separately when a checkpoint is gated.

For a small *trained* model, run the CPU pilot instead:

```bash
python -m repro plan configs/local_pilot_cpu.json
python -m repro run configs/local_pilot_cpu.json --output runs/my_cpu_pilot
python -m repro analyze runs/my_cpu_pilot --output runs/my_cpu_pilot_analysis
```

That pilot uses SmolLM2-135M, five questions, two seeds, and three noise levels. Its results are illustrative only, not a paper replication. Model downloads and the first setup may take time.

## Running a GPU experiment

The supplied RTX 4050 has 6 GB VRAM; it can run reduced pilots but does not fit the smallest full paper model in the configured precision. Cloud GPUs can be used for larger experiments. The estimates below are planning guidance, **not measured allocations**:

| Experiment | Suggested starting hardware |
| --- | --- |
| 135M pilot | 6 GB GPU or CPU; 16 GB host RAM |
| 7B–8B bf16 multiple-choice | 24 GB GPU may fit; A100 40 GB recommended, with around 64 GB host RAM |
| Phi-medium 14B bf16 | A100 40 GB as a practical starting point, 80 GB more comfortable; at least 64 GB host RAM |
| Mistral Large no-goal | Configured for 8 GPUs and substantial host RAM; see the detailed estimates |

See [hardware and dependencies](docs/EXPERIMENTS.md#hardware-and-dependencies), including CPU snapshot memory and separate Phi-small/vLLM environments. Do not silently introduce quantization or offloading and call the result the same intervention.

For an ordinary CUDA environment, install the CUDA-compatible PyTorch wheel in a separate environment, then the general requirements:

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -r requirements-analysis.txt
python scripts/prepare.py
```

For example, inspect a paper-scale prompted MCQ configuration before deciding to run it:

```bash
python -m repro plan configs/full/llama8b_ai2.json
python -m repro check configs/full/llama8b_ai2.json --output runs/llama_preflight.json
python -m repro run configs/full/llama8b_ai2.json --output runs/full/llama8b_ai2
python -m repro analyze runs/full/llama8b_ai2 --output runs/llama_ai2_analysis
```

This example is computationally expensive and requires access to the gated model. `python -m repro matrix` prints the full MCQ matrix; it **does not** run it. The model will download on first use. GPU environments and full runs remain unverified here, so perform a preflight and a reduced pilot first, and address the known issues before launching large campaigns. For Colab or other temporary machines, place run outputs and caches on persistent storage. Resume accepts an existing directory only if the recorded code, configuration, model/data identity, and environment still match.

The extended vLLM and Phi-small paths have separate, potentially incompatible requirement files; see [the other experiment commands](docs/EXPERIMENTS.md#running-the-other-experiments). Their settings are reconstructions where noted, not proof of historical fidelity.

## Understanding outputs

Each `run` output contains `metadata.json` (configuration, revisions, dependencies), `inputs.json` (formatted prompts and token IDs), `parameter_layout.json` (selected parameter metadata), per-trial JSON files with individual predictions, and either `completion.json` or `failure.json`. A failed question stops the trial rather than being dropped from the denominator.

An `analyze` output contains:

- `predictions.csv`: each question's output and score.
- `accuracy.csv`: accuracy by condition, noise magnitude, and seed.
- `measurements.csv` and `summary.csv`: baseline, maximum, improvement ratio, and summaries.
- `welch_bh.csv` and `report.json`: available statistical results and metadata.
- PNG figures: saved accuracy curves generated from the actual input results.

Confidence intervals and hypothesis tests require suitable independent measurements; repeated deterministic baselines are not independent samples. A `null` statistic may indicate insufficient measurements or an undefined ratio, not a zero effect. Original source-faithful scores and corrected diagnostics must be labeled separately.

The existing [CPU pilot report](results/final_cpu_pilot_analysis/report.json) is a useful example: its ordinary condition improved from 2/5 to 3/5 correct, and prompted sandbagging from 1/5 to 2/5. Those counts are too small for a research conclusion.

## Reanalyze the authors' published results

After preparing the pinned sources and installing the analysis dependencies:

```bash
python -m repro analyze --archive --classifier --output runs/author_archive_analysis
python scripts/compare_paper.py
```

Archive analysis reads published CSVs; it does not execute models. The comparison script uses the project's expected archive-analysis location; check its inputs before using it with a different output directory. See [RESULTS.md](docs/RESULTS.md) for the actual comparison and unresolved discrepancies.

## Sources and attribution

- [Paper, arXiv v3](https://arxiv.org/abs/2412.01784v3)
- [SandbagDetect](https://github.com/camtice/SandbagDetect)
- [noise_injection_extended](https://github.com/camtice/noise_injection_extended)

The repositories and their pinned revisions are recorded in [sources.lock.json](assets/sources.lock.json). This project is a research reproduction, not an official author implementation. Consult [ATTRIBUTION.md](ATTRIBUTION.md) and upstream licenses before redistributing third-party materials.
