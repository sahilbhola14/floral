# scripts/oneDCorr/print_wandb_stats.py
"""
Fetch and print training + inference summary stats from wandb projects:
  - onedcorr_flora   (FNO / deterministic operator runs)
  - onedcorr_floral  (FM / flow-matching runs)
"""
import wandb
import pandas as pd

# ---- user input ----
# set to "your-username" or "your-team"; None = wandb default entity
ENTITY = "sahilbhola14-university-of-michigan"
PROJECTS = ["onedcorr_flora", "onedcorr_floral"]

SUMMARY_KEYS = [
    # training cost
    "train_time_s",
    "time_per_epoch_s",
    "n_epochs_trained",
    "n_train_samples",
    "peak_gpu_memory_mb",
    # validation
    "best_val_loss",
    "final_val_loss",
    "best_epoch",
    # model size
    "n_params_total",
    "n_params_trainable",
]

CONFIG_KEYS = [
    "floral",
    "train.train_res",
    "train.objective",
]
# ---- end user input ----


def _deep_get(d: dict, dotted_key: str):
    """Traverse a nested dict using a dot-separated key path."""
    for part in dotted_key.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def get_runs(project: str) -> list:
    api = wandb.Api()
    path = f"{ENTITY}/{project}" if ENTITY else project
    return list(api.runs(path))


def extract_row(run, project: str) -> dict:
    row = {
        "project": project,
        "group": run.group,
        "run_name": run.name,
        "run_id": run.id,
        "state": run.state,
    }

    # summary metrics
    for key in SUMMARY_KEYS:
        row[key] = run.summary.get(key)

    # config values — traverse dot-notation keys into nested config dict
    cfg = run.config.get("config") or run.config
    for key in CONFIG_KEYS:
        row[key] = _deep_get(cfg, key)
    return row


def sort_runs(
    df: pd.DataFrame, keys: str | list, ascending: bool | list = True
) -> pd.DataFrame:
    keys = [keys] if isinstance(keys, str) else keys
    missing = [k for k in keys if k not in df.columns]
    if missing:
        print(
            f"Warning: sort key(s) {missing} not found in columns: {list(df.columns)}"
        )
        keys = [k for k in keys if k in df.columns]
    if not keys:
        return df
    if isinstance(ascending, bool):
        ascending = [ascending] * len(keys)
    return df.sort_values(by=keys, ascending=ascending, na_position="last").reset_index(
        drop=True
    )


def print_table(df: pd.DataFrame, title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))


def main():
    all_rows = []
    for project in PROJECTS:
        print(f"Fetching runs from: {project} ...")
        runs = get_runs(project)
        print(f"  {len(runs)} runs found")
        for run in runs:
            all_rows.append(extract_row(run, project))

    if not all_rows:
        print("No runs found.")
        return

    df = pd.DataFrame(all_rows)

    # --- full summary ---
    display_cols = [
        "floral",
        # "group",
        # "train.train_res",
        "train.objective",
        "n_train_samples",
        "train_time_s",
        "peak_gpu_memory_mb",
        "n_params_trainable",
        # "n_params_total",
    ]
    print_table(
        df[[c for c in display_cols if c in df.columns]], "All runs — training summary"
    )

    # --- sort based on n_train_samples ---
    df_sorted = sort_runs(df, keys=["n_train_samples", "floral"])
    print_table(
        df_sorted[[c for c in display_cols if c in df_sorted.columns]],
        "All runs — sorted by n_train_samples",
    )


if __name__ == "__main__":
    main()
