# Reproducing the ablation tables

Trained checkpoints are **not distributed** (hundreds of GB); the tables are reproduced by
training the sweep, then scoring it. Every table in the report renders from one CSV.

## 1. Train the sweep

Each sweep's flags live in `sweep_configs/*.yaml` — a `common` flag string plus one `{name, args}`
per run, passed verbatim to its trainer (`train_script`, default `experiments/ksa_aligned_rdn_train.py`;
the WA and U-Net configs point at their own trainers). `train.py` deploys them:

```bash
uv run python experiments/ksa_ablations/train.py --all         # print the 62 train commands
uv run python experiments/ksa_ablations/train.py --all --run   # launch them (-> logs/<name>.log)
```

Prerequisites: the patch indices from `scripts/build_ksa_dataset/` (base fold splits +
`build_dense_s12_cv_folds.sh` for the stride-12 folds; the synuc grid additionally needs the
p264 repatch from `05b_repatch_fixed_cells.py`) and the WA patches from `scripts/build_wa_dataset/`.
Runs listed in two sweeps (the shared nb13 controls) are trained once — `train.py` dedupes.

## 2. Score everything -> one CSV -> LaTeX

```bash
uv run python experiments/ksa_ablations/evaluate.py     # score every checkpoint -> ONE csv
uv run python experiments/ksa_ablations/make_tables.py --out experiments/ksa_ablations/tables_generated.tex
```

`evaluate.py` writes `results/ksa_ablations_eval.csv` with three scorers, one command:
real-LR model runs via `sweep_all_models` (each on its own fold/dataset), bicubic via the shared
`evaluate_loader`, and the 12 synthetic-operator runs via `eval_synuc.score_synuc` (seeded synthetic
pairs, the wrong task for the real-LR sweep). `--skip-sweep` / `--skip-synuc` reuse existing rows.
The sweep also builds each run's missing `_soup_ssim_rmse.pt` (`magsr.models.ensure_soup`, the 50/50
average of `_best` and `_best_rmse`) so every run offers the same three candidate checkpoints.

`make_tables.py` renders every table from that one CSV (`latex_tables.yaml` is the run list): it selects
each run's checkpoint by argmin validation RMSE, bolds the lowest test RMSE, and appends the reference
baselines. The exp_ideal table computes coupling-recovered from the flat/drape mag-only floors.

The report's training-configuration table is hand-authored from the trainer defaults and these
sweep configs; it has no generator.
