"""Command-line entry points; heavy ML imports stay inside the run command."""
import argparse
import json
from pathlib import Path
from .config import load_config, load_questions, magnitudes, validate
from .io import ROOT, read_json


def main():
    parser = argparse.ArgumentParser(description="Source-audited sandbagging noise reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["plan", "run", "check"]:
        command = commands.add_parser(name)
        command.add_argument("config")
        if name in {"run", "check"}:
            command.add_argument("--output", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("directories", nargs="*")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--archive", action="store_true")
    analyze.add_argument("--classifier", action="store_true")
    analyze.add_argument("--corrected", action="store_true")
    matrix = commands.add_parser("matrix")
    matrix.add_argument("config", default="configs/main_matrix.json", nargs="?")
    scaling = commands.add_parser("scaling")
    scaling.add_argument("directory")
    baseline = scaling.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline", type=float)
    baseline.add_argument("--baseline-run")
    scaling.add_argument("--output", required=True)
    scaling.add_argument("--with-replacement", action="store_true")
    args = parser.parse_args()
    if args.command in {"plan", "run", "check"}:
        config = validate(load_config(args.config))
        if args.command == "plan":
            from .prompts import conditions
            seeds = sum(len(s) for s in config["seed_groups"].values())
            count = config["data"].get("limit") or config["data"].get("expected_count")
            print(json.dumps({"config": config, "conditions": conditions(config), "noise_levels": len(magnitudes(config)),
                              "measurement_count": len(config["seed_groups"]), "noise_seed_count": seeds,
                              "paired_trials": seeds * len(magnitudes(config)),
                              "question_condition_evaluations": seeds * len(magnitudes(config)) * count * len(conditions(config))}, indent=2))
        elif args.command == "check":
            from .checks import preflight
            print(json.dumps(preflight(config, args.output), indent=2))
        else:
            from .runner import run
            run(config, args.output)
    elif args.command == "analyze":
        from .analysis import analyze
        result = analyze(args.directories, args.output, args.archive, args.classifier, args.corrected)
        print(f"Saved analysis: {args.output}; {result['origin']}; {result['n_accuracy_rows']} accuracy rows")
    elif args.command == "matrix":
        for job in read_json(args.config)["jobs"]:
            config = validate(load_config(ROOT / job))
            print(f"python -m repro run {job} --output runs/full/{config['name']}")
    elif args.command == "scaling":
        from .analysis import scaling_report, run_rows
        value=args.baseline
        if args.baseline_run:
            rows,_=run_rows(args.baseline_run)
            value=max(r["accuracy"] for r in rows if r["magnitude"]==0 and r["condition"] in {"ordinary","password_present"})
        if value <= 0:
            raise ValueError("Baseline must be positive")
        scaling_report(args.directory, value, args.output, args.with_replacement)


if __name__ == "__main__":
    main()
