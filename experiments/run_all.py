from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full publication pipeline: collect -> visualize -> run-safety-pipeline."
    )
    parser.add_argument("--model", default="gemma3_4b")
    parser.add_argument("--dataset", default="toy_contrastive")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--activations-root", type=Path, default=Path("activations"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/outputs"))
    parser.add_argument("--shifted-activations", type=Path, default=None)
    parser.add_argument("--real-sap", action="store_true")
    parser.add_argument("--sap-repo-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    collect_cmd = [
        python,
        "scripts/collect_trajectories.py",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--num_samples",
        str(args.num_samples),
        "--output",
        str(args.activations_root),
    ]
    if args.layer is not None:
        collect_cmd.extend(["--layers", str(args.layer)])
    _run(collect_cmd, repo_root)

    # Infer layer path if layer not supplied by using defaults in config.
    if args.layer is None:
        default_layer = {"llama_3_1_8b": 20, "qwen3_8b": 24, "gemma3_4b": 18}.get(args.model, 5)
    else:
        default_layer = args.layer
    activations_leaf = args.activations_root / args.dataset / "train" / args.model / f"layer_{default_layer}"

    viz_cmd = [
        python,
        "scripts/visualize_trajectories.py",
        "--model",
        args.model,
        "--activations",
        str(activations_leaf),
        "--fig_dir",
        str(args.fig_dir),
    ]
    _run(viz_cmd, repo_root)

    safety_cmd = [
        python,
        "-m",
        "latent_dynamics",
        "run-safety-pipeline",
        "--activations",
        str(activations_leaf),
        "--output-dir",
        str(args.output_dir),
        "--plot-drift",
        "--seed",
        str(args.seed),
    ]
    if args.shifted_activations is not None:
        safety_cmd.extend(["--shifted-activations", str(args.shifted_activations)])
    if args.real_sap:
        safety_cmd.append("--real-sap")
    if args.sap_repo_path is not None:
        safety_cmd.extend(["--sap-repo-path", str(args.sap_repo_path)])
    _run(safety_cmd, repo_root)

    print("run_all finished successfully.")


if __name__ == "__main__":
    main()

