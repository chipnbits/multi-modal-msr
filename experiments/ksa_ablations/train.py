"""Deploy training for one sweep config (or all of them).

Evaluate AFTER training, as a separate step, then rebuild the tables:
    uv run python experiments/ksa_ablations/evaluate.py        # score all checkpoints -> CSV
    uv run python experiments/ksa_ablations/make_tables.py     # CSV -> LaTeX

    uv run python experiments/ksa_ablations/train.py sweep_configs/modality.yaml         # print commands
    uv run python experiments/ksa_ablations/train.py sweep_configs/modality.yaml --run   # run them (-> logs/)
    uv run python experiments/ksa_ablations/train.py --all | bash                        # run every sweep
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml

from magsr import ROOT_FOLDER

SWEEP_CONFIG_DIR = ROOT_FOLDER / "experiments/ksa_ablations/sweep_configs"
RDN_TRAINER = "experiments/ksa_aligned_rdn_train.py"


def sweep_commands(path: Path):
    """Yield (name, argv) per run: <launcher> <trainer> --run-name <name> <common> <args>."""
    cfg = yaml.safe_load(open(path))
    common = cfg.get("common", "")
    launch = ["python"] if cfg.get("launcher") == "python" else ["accelerate", "launch"]
    for run in cfg["runs"]:
        trainer = run.get("train_script") or cfg.get("train_script", RDN_TRAINER)
        argv = [
            *launch,
            trainer,
            "--run-name",
            run["name"],
            *shlex.split(common),
            *shlex.split(run.get("args", "")),
        ]
        yield run["name"], argv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, nargs="?", help="a sweep config YAML")
    ap.add_argument("--all", action="store_true", help="every sweep_configs/*.yaml")
    ap.add_argument("--run", action="store_true", help="launch (default: print); logs to logs/<name>.log")
    args = ap.parse_args()

    if args.all:
        configs = sorted(SWEEP_CONFIG_DIR.glob("*.yaml"))
    elif args.config:
        configs = [args.config]
    else:
        ap.error("give a config path or --all")

    seen: set[str] = set()  # two sweeps list the same nb13 runs; train each once
    for c in configs:
        for name, cmd in sweep_commands(c):
            if name in seen:
                continue
            seen.add(name)
            if not args.run:
                print("uv run " + " ".join(cmd))
                continue
            (ROOT_FOLDER / "logs").mkdir(exist_ok=True)
            print(f"=== training {name}")
            with open(ROOT_FOLDER / f"logs/{name}.log", "w") as log:
                subprocess.run(
                    ["uv", "run", *cmd], cwd=ROOT_FOLDER, check=True, stdout=log, stderr=subprocess.STDOUT
                )


if __name__ == "__main__":
    main()
