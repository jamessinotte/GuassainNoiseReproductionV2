# Honest results record

All implementation and outputs are in V2. No implementation was copied from or written to the first reproduction folder.

## Executed locally

| Check / run | Actual outcome | Evidence |
| --- | --- | --- |
| Focused automated suite | 25 tests passed; no skipped tests | [validation record](../results/validation/record.json), [test output](../results/validation/tests.log) |
| Dependency consistency / compilation | `pip check` passed; Python compilation passed | [dependency output](../results/validation/dependencies.log), [resolved dependency versions](../results/validation/requirements-resolved.txt) |
| Tiny real-checkpoint preflight | Zero-noise prediction unchanged; fixed-seed perturbation deterministic; exact bitwise restoration; restored prediction unchanged | [checkpoint preflight](../results/validation/checkpoint_preflight.json) |
| Random 1M-parameter Llama smoke | Six paired trials, three questions, two conditions, 36 saved predictions; 0% accuracy; phi undefined | [validated smoke run](../results/validated_smoke/), [analysis](../results/final_smoke_analysis/) |
| Trained SmolLM2-135M CPU pilot | Six paired trials, first five AI2 questions, two conditions, 60 saved predictions; all trials completed | [validated CPU pilot](../results/validated_cpu_pilot/), [analysis](../results/final_cpu_pilot_analysis/) |
| Author CSV reanalysis | 59,000 original accuracy rows, 80 versioned files, 16 model/benchmark combinations; combined duplicates excluded | [archive report](../results/author_archive/report.json), [paper comparison](../results/author_archive/paper_comparison.csv) |
| Full Llama-8B run attempt | Stopped before model download/inference because CUDA is unavailable | [failure record](../results/full_run_attempt/failure.json) |
| Extended dataset preparation | Ordered first 30 GSM8K test questions saved with immutable dataset revision and content hash | [dataset provenance](../data/gsm8k_provenance.json) |

The tests cover zero-noise weights **and logits**, fp32/bf16/fp16 restoration, the source per-tensor seed schedule, independence from global RNG, absolute scale, selected parameter coverage, PEFT's nonzero A/zero B initialization, exact prompt literals/password placement, strict next-token scoring, known parser bugs and corrected diagnostics, real tiny-transformer forward passes, paired weights, resume identity/checksums, measurement aggregation, duplicate-seed independence checks, and dropped-question rejection. They do not certify unavailable CUDA/Flash-Attention/vLLM paths.

## Small trained pilot: illustrative, not a paper replication

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, resolved revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. CPU float32; first five questions; seeds 11 and 22; absolute noise magnitudes 0, .001, .003; published Llama-style prompts with this model's tokenizer. This is **one measurement** with two seeds, not five independent measurements.

| Condition | Clean accuracy | Maximum observed accuracy | phi |
| --- | --- | --- | --- |
| Ordinary | .40 | .60 | 1.50 |
| Prompted sandbagging | .20 | .40 | 2.00 |

These are actual saved predictions, but five questions are far too few for a research conclusion. The ordinary model also improves by one question under noise. No confidence interval, significance claim, trained detector claim, or paper-level generalization is inferred from this pilot. It demonstrates that the runnable inference/noise/scoring pipeline works on a small trained checkpoint. The random smoke's all-wrong answers are not counted as successful sandbagging detection.

## Comparison to the published results

Recomputing the maximum-over-five-seeds measurement statistic from the **authors' archives** recovers all **30 prompted condition means** (15 model/benchmark pairs × two conditions) within v3 Table 1's displayed rounding. Example: Llama-8B/AI2 gives standard 1.0297297 and prompted-sandbagging 1.9741935, displayed as 1.030 and 1.974 in the paper. This validates the statistical interpretation/data reader, **not new model inference or historical prompt/checkpoint fidelity**.

The password-labeled archive does **not** recover v3's password result: its historical index-0/index-1 means are approximately 1.04494 and 1.09538, whereas v3 reports 1.01 and 3.12. The archived model ID/prompt provenance is inconsistent with the current adapter loader; those index labels are not treated as verified password states. This discrepancy is retained, not repaired by changing labels, searching for favorable subsets, or fabricating results.

The explicit paper-style classifier on the available 16-combination archive yields macro model-held-out accuracy **0.93333**. This is not the paper's **0.962**: the archive collection lacks the complete v3 math/no-goal/holdout observations, its password result differs, and unspecified XGBoost settings were reconstructed explicitly. The [machine-readable report](../results/author_archive/report.json) includes folds, sample counts, predictions and metrics; it does not claim an exact detector reproduction. Welch/BH outputs use the supplied 16-comparison family, not the complete v3 family. A SciPy precision-loss warning occurs for nearly constant measurement values; the inputs and raw/adjusted p-values remain exposed.

## Unexecuted / unresolved

- No new full 3.8B/7B/8B/14B sweeps, password-model runs, vLLM runs, 100-seed runs, or Mistral-Large no-goal runs have executed. The current session has no working NVIDIA driver; the specified RTX 4050 also cannot hold the paper's smallest full bf16 model in 6 GB.
- Phi-small's remote Triton/Flash-Attention compatibility profile is provided but not GPU-tested. The pinned vLLM profile has not been installed or exercised locally.
- Original historical checkpoint revisions, some task/sweep/seed details, exact no-goal input whitespace and WMDP subset preparation remain unavailable. Configurations label reconstruction choices; the 70B quantized intervention is unresolved rather than silently replaced.
- The two repositories disagree with the paper and each other on prompts, password grids, model IDs, task sizes and noise units. The [specification](EXPERIMENTS.md) maps every choice and separates faithful source behavior from corrected/portable interpretations.
- The old legacy CoT invocation is broken and its parser can score prompt numbers. Its available scorer/stop behavior is documented; current extended mathematical tasks are separate experiments, not relabeled exact old-CoT reproductions.

## Ready-to-run commands

The [README](../README.md) gives setup/smoke/main-matrix commands. The [full specification](EXPERIMENTS.md#running-the-other-experiments) gives password, extended and scaling commands and appropriate hardware. Exact immutable models, configurations, prompt assets and dataset hashes are saved. A failed/incomplete experiment never becomes a completed result or silently changes the denominator.

Earlier development smoke/pilot outputs are retained separately from the `validated_*` runs; strict resume rejects code/environment changes. Each record identifies the code/config/environment at execution, so those earlier runs are not additional independent measurements. The `final_*_analysis` directories analyze the validated runs.
