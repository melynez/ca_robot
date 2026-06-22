from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# =========================================================
# CONFIG
# =========================================================

FIRST_POST_BARRIER_GEN = 50
RULES = list(range(256))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regular_original_dir", required=True)
    ap.add_argument("--regular_barrier_dir", required=True)
    ap.add_argument("--frameshift_original_dir", default=None)
    ap.add_argument("--frameshift_barrier_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--first_post_barrier_gen", type=int, default=50)
    return ap.parse_args()

# =========================================================
# HELPERS
# =========================================================

def safe_read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"bin_string": str})


def unify_gen_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert either:
        generation -> gen
    or keep gen
    """
    df = df.copy()

    if "gen" in df.columns:
        return df

    if "generation" in df.columns:
        df = df.rename(columns={"generation": "gen"})
        return df

    raise KeyError(
        f"No generation column found. Columns are:\n{df.columns.tolist()}"
    )


def prepare_original(df_original: pd.DataFrame) -> pd.DataFrame:
    df = unify_gen_column(df_original)

    if "bin_string" not in df.columns:
        raise KeyError(
            f"'bin_string' missing in original file.\nColumns:\n{df.columns.tolist()}"
        )

    df["gen"] = pd.to_numeric(df["gen"], errors="coerce")
    df["bin_string"] = df["bin_string"].fillna("").astype(str)

    df = df[df["gen"] >= FIRST_POST_BARRIER_GEN].copy()
    df["original_length"] = df["bin_string"].str.len()

    return df[["gen", "original_length"]]


def prepare_barrier(df_barrier: pd.DataFrame) -> pd.DataFrame:
    df = unify_gen_column(df_barrier)

    df["gen"] = pd.to_numeric(df["gen"], errors="coerce")

    # if explicit length exists use it,
    # otherwise compute from bin_string
    if "length_after" in df.columns:
        df["length_after"] = (
            pd.to_numeric(df["length_after"], errors="coerce")
            .fillna(0)
        )
    else:
        if "bin_string" not in df.columns:
            raise KeyError(
                f"No length_after or bin_string in barrier file.\nColumns:\n{df.columns.tolist()}"
            )

        df["bin_string"] = df["bin_string"].fillna("").astype(str)
        df["length_after"] = df["bin_string"].str.len()

    # extinction optional
    if "extinction" in df.columns:
        df["extinction"] = (
            df["extinction"]
            .fillna(False)
            .astype(bool)
        )
    else:
        df["extinction"] = False

    df = df[df["gen"] >= FIRST_POST_BARRIER_GEN].copy()

    df = (
        df.sort_values("gen")
        .drop_duplicates(subset=["gen"], keep="last")
    )

    return df[["gen", "length_after", "extinction"]]


def compute_bri(
    df_original,
    df_barrier,
    rule,
    dataset_name,
    original_file,
    barrier_file,
):
    orig = prepare_original(df_original)
    barr = prepare_barrier(df_barrier)

    merged = (
        orig.merge(
            barr,
            on="gen",
            how="left"
        )
        .sort_values("gen")
    )

    # generations after extinction = zero retained bits
    merged["length_after"] = (
        pd.to_numeric(merged["length_after"], errors="coerce")
        .fillna(0)
    )

    n_theoretical = float(
        merged["original_length"].sum()
    )

    n_actual = float(
        merged["length_after"].sum()
    )

    bri = (
        n_actual / n_theoretical
        if n_theoretical > 0
        else np.nan
    )

    extinct_rows = barr[barr["extinction"] == True]

    extinction = len(extinct_rows) > 0

    extinction_gen = (
        float(extinct_rows["gen"].min())
        if extinction
        else np.nan
    )

    return {
        "dataset": dataset_name,
        "rule": rule,
        "first_post_barrier_gen": FIRST_POST_BARRIER_GEN,
        "bri": bri,
        "n_theoretical": n_theoretical,
        "n_actual": n_actual,
        "post_barrier_gens_expected": int(len(merged)),
        "post_barrier_gens_with_nonzero_retention": int(
            (merged["length_after"] > 0).sum()
        ),
        "extinction": extinction,
        "extinction_gen": extinction_gen,
        "original_file": str(original_file),
        "barrier_file": str(barrier_file),
    }


def process_dataset(dataset_name, cfg):
    rows = []
    missing = []

    print("\n" + "=" * 70)
    print("PROCESSING:", dataset_name)
    print("=" * 70)

    print("Original dir exists:", cfg["original_dir"].exists())
    print("Barrier dir exists :", cfg["barrier_dir"].exists())

    print(
        "Original CSV count:",
        len(list(cfg["original_dir"].glob("*.csv")))
    )

    print(
        "Barrier CSV count:",
        len(list(cfg["barrier_dir"].glob("*.csv")))
    )

    for rule in RULES:

        original_file = (
            cfg["original_dir"]
            / f"dfOriginal_r{rule}.csv"
        )

        barrier_file = (
            cfg["barrier_dir"]
            / f"dfBarrier_r{rule}.csv"
        )

        df_original = safe_read_csv(original_file)
        df_barrier = safe_read_csv(barrier_file)

        if df_original is None:
            missing.append({
                "dataset": dataset_name,
                "rule": rule,
                "problem": "missing_original",
                "path": str(original_file),
            })
            continue

        if df_barrier is None:
            missing.append({
                "dataset": dataset_name,
                "rule": rule,
                "problem": "missing_barrier",
                "path": str(barrier_file),
            })
            continue

        try:
            row = compute_bri(
                df_original=df_original,
                df_barrier=df_barrier,
                rule=rule,
                dataset_name=dataset_name,
                original_file=original_file,
                barrier_file=barrier_file,
            )

            rows.append(row)

        except Exception as e:

            missing.append({
                "dataset": dataset_name,
                "rule": rule,
                "problem": "processing_error",
                "error": repr(e),
                "original_columns": str(df_original.columns.tolist()),
                "barrier_columns": str(df_barrier.columns.tolist()),
            })

            # print first few errors live
            if len(missing) <= 5:
                print(f"\nERROR rule {rule}")
                print(repr(e))
                print("original columns:", df_original.columns.tolist())
                print("barrier columns :", df_barrier.columns.tolist())

    df_metrics = pd.DataFrame(rows)

    if not df_metrics.empty:
        df_metrics = df_metrics.sort_values(
            ["dataset", "rule"]
        )
    else:
        print(f"\nWARNING: no rows created for {dataset_name}")

    df_missing = pd.DataFrame(missing)

    df_metrics.to_csv(
        cfg["output_csv"],
        index=False
    )

    df_missing.to_csv(
        cfg["missing_csv"],
        index=False
    )

    print("\nSaved:")
    print(cfg["output_csv"])
    print(cfg["missing_csv"])
    print("rows =", len(df_metrics))
    print("missing/errors =", len(df_missing))

    return df_metrics


def main():

    args = parse_args()

    OUTPUT_DIR = Path(args.out_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    FIRST_POST_BARRIER_GEN = args.first_post_barrier_gen

    DATASETS = {
        "ca_regular": {
            "original_dir": Path(args.regular_original_dir),
            "barrier_dir": Path(args.regular_barrier_dir),
            "output_csv": OUTPUT_DIR / "metrics_bri_ca_regular.csv",
            "missing_csv": OUTPUT_DIR / "metrics_bri_ca_regular_missing.csv",
        }
    }

    if args.frameshift_original_dir and args.frameshift_barrier_dir:
        DATASETS["ca_frameshift"] = {
            "original_dir": Path(args.frameshift_original_dir),
            "barrier_dir": Path(args.frameshift_barrier_dir),
            "output_csv": OUTPUT_DIR / "metrics_bri_ca_frameshift.csv",
            "missing_csv": OUTPUT_DIR / "metrics_bri_ca_frameshift_missing.csv",
        }

if __name__ == "__main__":
    main()
    
    
